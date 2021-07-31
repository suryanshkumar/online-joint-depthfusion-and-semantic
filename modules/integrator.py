import torch
import numpy as np
import time

class Integrator(torch.nn.Module):

    def __init__(self, config):

        super(Integrator, self).__init__()

        self.config = config
        self.device = config.SETTINGS.device
        self.implementation = config.SETTINGS.implementation

    def forward(self, updates, values_volume, weights_volume, scores_volume, semantics_volume, test=True):
        """
            Calculate updates and integrate them in the respective volumes

            :param updates: list of updates [tsdf_values, indices, weights, semantics]
            :param values_volume: old tsdf volume from database (on cpu)
            :param weights_volume: old weights volume from database (on cpu)
            :param semantics_volume: old semantics volume from database (on cpu), 
                                     histograms in test mode,
                                     labels in train/val mode,
                                     None if semantics disabled
            :return: updated volumes
            """

        values = updates['values'].to(self.device)
        indices = updates['indices'].to(self.device)
        weights = updates['weights'].to(self.device)

        xs, ys, zs = values_volume.shape

        # reshape tensors
        n1, n2, n3 = values.shape

        indices = indices.contiguous().view(n1 * n2 * n3, 8, 3).long()
        weights = weights.contiguous().view(n1 * n2 * n3, 8)
        values = values.contiguous().view(n1 * n2 * n3, 1).repeat(1, 8)

        i1, i2, i3 = indices.shape

        indices = indices.contiguous().view(i1 * i2, i3).long()
        weights = weights.contiguous().view(i1 * i2, 1).float()
        values = values.contiguous().view(i1 * i2, 1).float()

        valid = get_index_mask(indices, values_volume.shape)
        valid_idx = torch.nonzero(valid)[:, 0]

        indices = extract_indices(indices, mask=valid)
        weights = torch.masked_select(weights[:, 0], valid)
        values = torch.masked_select(values[:, 0], valid)

        update = weights * values

        index = ys * zs * indices[:, 0] + zs * indices[:, 1] + indices[:, 2]
  
        cache = torch.zeros(xs * ys * zs, device=self.device).float()
        cache.index_add_(0, index, weights)
        cache = cache.view(xs, ys, zs)
        weights = extract_values(indices, cache)

        cache = torch.zeros(xs * ys * zs, device=self.device).float()
        cache.index_add_(0, index, update)
        cache = cache.view(xs, ys, zs)
        update = extract_values(indices, cache)

        del cache, index

        #indices = indices.to('cpu')
        weights_old = extract_values(indices, weights_volume)
        weights_old = weights_old.to(self.device).float()
        values_old = extract_values(indices, values_volume)
        values_old = values_old.to(self.device).float()

        weight_update = weights_old + weights
        weight_update = weight_update.half()
        if self.implementation != 'efficient':
            weight_update = weight_update.to('cpu')

        value_update = (weights_old * values_old + update) / (weights_old + weights)
        value_update = value_update.half()
        if self.implementation != 'efficient':
            value_update = value_update.to('cpu')

        insert_values(weight_update, indices, weights_volume)  #CPU
        insert_values(value_update, indices, values_volume)    #CPU

        if self.config.DATA.semantics and test:

            # use the new semantic values to update the volume
            ids = updates['semantics'].to(self.device)
            ids = ids.contiguous().view(n1 * n2 * n3, 1).repeat(1, 8)
            ids = ids.contiguous().view(i1 * i2)
            ids = ids[valid_idx]

            scores = updates['scores'].to(self.device)
            scores = scores.contiguous().view(n1 * n2 * n3, 1).repeat(1, 8)
            scores = scores.contiguous().view(i1 * i2)
            scores = scores[valid_idx]

            ids_old = extract_values(indices, semantics_volume)
            ids_old = ids_old.to(self.device)
            mask = ids_old != ids

            scores_old = extract_values(indices, scores_volume)
            scores_old = scores_old.to(self.device).float()

            # score update:
            # ids == ids_old --> update if scores > scores_old
            # ids != ids_old --> update if scores > scores_old
            scores_update = torch.where(scores > scores_old, scores, scores_old)
            scores_update = scores_update.half()

            semantic_update = torch.where(scores > scores_old, ids, ids_old)
            semantic_update = semantic_update[mask]

            if self.implementation != 'efficient':
                semantic_update = semantic_update.to('cpu')
                scores_update = scores_update.to('cpu')

            insert_values(semantic_update, indices[mask], semantics_volume)
            insert_values(scores_update, indices, scores_volume)

        return values_volume, weights_volume, semantics_volume, scores_volume


def get_index_mask(indices, shape):
    """
    method to check whether indices are valid
    :param indices: indices to check
    :param shape: constraints for indices
    :return: mask
    """
    xs, ys, zs = shape

    valid = ((indices[:, 0] >= 0) &
             (indices[:, 0] < xs) &
             (indices[:, 1] >= 0) &
             (indices[:, 1] < ys) &
             (indices[:, 2] >= 0) &
             (indices[:, 2] < zs))

    return valid


def extract_values(indices, volume, mask=None):
    """
    method to extract values from volume given indices
    :param indices: positions to extract
    :param volume: volume to extract from
    :param mask: optional mask for extraction
    :return: extracted values
    """
    if mask is not None:
        x = torch.masked_select(indices[:, 0], mask)
        y = torch.masked_select(indices[:, 1], mask)
        z = torch.masked_select(indices[:, 2], mask)
    else:
        x = indices[:, 0]
        y = indices[:, 1]
        z = indices[:, 2]
    return volume[x, y, z]


def extract_indices(indices, mask):
    """
    method to extract indices according to mask
    :param indices:
    :param mask:
    :return:
    """

    x = torch.masked_select(indices[:, 0], mask)
    y = torch.masked_select(indices[:, 1], mask)
    z = torch.masked_select(indices[:, 2], mask)

    masked_indices = torch.cat((x.unsqueeze_(1),
                                y.unsqueeze_(1),
                                z.unsqueeze_(1)), dim=1)
    return masked_indices


def insert_values(values, indices, volume):
    """
    method to insert values back into volume
    :param values:
    :param indices:
    :param volume:
    :return:
    """
    # print(volume.dtype)
    # print(values.dtype)
    # volume = volume.half()
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = values

    #return volume
