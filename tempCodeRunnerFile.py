 def visualize_model(self):
        """
        Graphyical Visualize the model architecture using visualkeras.
        """
        img = visualkeras.layered_view(self.model, to_file='model_visual.png')  
        img.show() 