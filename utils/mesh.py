import numpy as np
from plyfile import PlyElement, PlyData

#==============================================================================
#                              PlyElement wrapper
#==============================================================================

class PlyElementPlus():
	def __init__(self, element):
		if not isinstance(element, PlyElement):
			TypeError("Expected PlyElement as argument")
		self.element = element
		self.count = element.count
		self.name = element.name
		self.comments = element.comments
		self.properties = list(element.data.dtype.names)
		self.types = [str(element.data[n].dtype) for n in self.properties]
		self._shapes = [self.__get_shape(element.data[n]) for n in self.properties]

		# When reading mesh, structured arrays as ('name', 'i4', (3,)) becomes of type 'object' --> errors in plyfile.
		# Thus, simply change type to the element type
		for i in range(len(self.types)):
			if self.types[i] == 'object' and self._shapes[i] > 1:
				self.types[i] = self.element[self.properties[i]][0].dtype

	def remove_property(self, name):
		"""
		Remove queried property by the PlyElement
		"""
		if len(self.properties) == 1:
			self.reset()
			return

		idx = self.properties.index(name)
		
		idx_ = np.arange(idx, idx + self._shapes[idx])
		element = [tuple(np.delete(np.array(list(el), dtype=object), idx_)) for el in self.element]

		self.properties.pop(idx)
		self.types.pop(idx)
		self._shapes.pop(idx)
		self.element = self._describe(element, name=self.name, dtype=self.__get_dtype_struct())

	def append_property(self, name, dtype, data=None):
		"""
		Append property at the end of the input PlyElement.
		If no data provided, add fake elements
		"""
		if name in self.properties:
			self.remove_property(name)

		if not self.element:
			if data is None:
				ValueError("Argument 'data' cannot be None if the object is empty.")
			self.count = len(data)

		if data is None:
			data = np.zeros(self.count)

		self.properties.append(name)
		self.types.append(dtype)
		self._shapes.append(self.__get_shape(data))

		if self.element:
			element = [(tuple(self.element[i]) + (data[i],)) for i in range(self.count)]
		else:
			element = [((data[i],)) for i in range(self.count)]

		self.element = self._describe(element, name=self.name, dtype=self.__get_dtype_struct())

	def reset(self):
		self.element = []
		self.properties = []
		self.types = []
		self._shapes = []
		self.count = 0		

	def refresh(self, data):
		"""
		Overwrite old object with new values of all properties
		"""
		if not isinstance(data, list):
			print("Expected data as list.")
		dtype = self.element.dtype()
		element = [tuple(row) for row in zip(*data)]
		element = np.array(element, dtype=dtype)
		self.element = self.element.describe(element, self.name)

	@staticmethod
	def _describe(data, name, dtype, **kwargs):
		"""
		data = np.array
		name = string
		dtypes = list of tuples (name, dtype, shape)
		"""
		try: data = np.array([(d,) for d in data], dtype=dtype)
		except: data = np.array([tuple(d) for d in data], dtype=dtype)
		return PlyElement.describe(data, name, **kwargs)

	def __get_shape(self, data):
		return len(data[0]) if isinstance(data[0], np.ndarray) else 1

	def __get_dtype_struct(self):
		return [(n, t) if s == 1 else (n, t, (s,)) 
				for (n, t, s) in zip(self.properties, self.types, self._shapes)]

	def __getitem__(self, key):
		return self.element.data[key]

	def __setitem__(self, key, value):
		self.element.data[key] = value

	def __str__(self):
		return self.element.header

	def __repr__(self):
		return ('PlyElementPlus(%r, %r, count=%d, comments=%r)' %
				(self.name, self.properties, self.count,
				 self.comments))


class Vertex(PlyElementPlus):
	def __init__(self, element):
		super().__init__(element)
		if not element.name == 'vertex':
			print("Wrong class, you may be after Face object.")
	
	@staticmethod
	def describe(*args, **kwargs):
		return Vertex(PlyElementPlus._describe(*args, **kwargs))


class Face(PlyElementPlus):
	def __init__(self, element):
		super().__init__(element)
		if not element.name == 'face':
			print("Wrong class, you may be after Vertex object.")
		if len(element['vertex_indices'][0]) > 3:
			self.quadmesh_to_trimesh()

	@staticmethod
	def describe(*args, **kwargs):
		return Face(PlyElementPlus._describe(*args, **kwargs))

	def quadmesh_to_trimesh(self):
		"""
		Transform quadmesh to trimesh
		"""
		quad_vertex_indices = np.vstack(self.element['vertex_indices'])
		quad_labels = np.array(self.element['object_id']) 
		tri_vertex_indices, tri_labels = quad_to_tri(quad_vertex_indices, quad_labels)
		self.refresh([tri_vertex_indices, tri_labels])


#==============================================================================
#                       Transform quadmesh into trimesh
#==============================================================================


def quad_to_tri(quad_vertices, quad_labels):
	"""
	[0,1,2,3] -> [0,1,2], [0,2,3]
	[0,1,2,1] -> [0,1,2]
	"""
	tri_vertices = np.empty((quad_vertices.shape[0],2,3), dtype=quad_vertices.dtype)      
	tri_vertices[:,0,1:] = quad_vertices[:,1:-1]
	tri_vertices[:,1,1:] = quad_vertices[:,2:]
	tri_vertices[...,0] = quad_vertices[:,0,None]
	tri_vertices.shape = (-1,3)

	tri_labels = np.array([val for pair in zip(quad_labels, quad_labels) for val in pair])

	mask = np.array([not((tri[0] == tri[1]) or (tri[0] == tri[2]) or (tri[1] == tri[2])) for tri in list(tri_vertices)])
	return list(tri_vertices[mask]), list(tri_labels[mask])

