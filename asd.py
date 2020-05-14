class Operation():
    def __init__(self,input_nodes=[]):
        self.input_nodes=input_nodes
        self.output_nodes=[]
        for node in input_nodes:
            node.output_nodes.append(self)
    def comute(self):
        pass
class add(Operation):
    def __init__(self,x,y):
        super().__init__([x,y])
    def comute(self,x_var,y_var): #Overwritting the comute functio up
        self.inputs=[x_var,y_var]
        return x_var+y_var

class multiply(Operation):
    def __init__(self,x,y):
        super().__init__([x,y])
        def comute(self,x_var,y_var):
            self.inputs=[x_var,y_var]
            return x_var*y_var

