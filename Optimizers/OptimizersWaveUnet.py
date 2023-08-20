from keras import optimizers

class CustomOptimizers():
	def __init__(self, learning_rate=0.01, **kwargs) -> None:
		self.learning_rate = learning_rate
	
	def __call__(self):
		return optimizers.Adam(learning_rate=self.learning_rate)