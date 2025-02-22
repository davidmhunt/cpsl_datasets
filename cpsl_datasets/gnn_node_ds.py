import os
import numpy as np
import matplotlib.image as img #not needed for ROS

class GnnNodeDS:

    def __init__(self,
                 dataset_path,
                 node_folder="nodes",
                 label_folder="labels",
                 ) -> None:
        

        #node folder
        self.nodes_enabled = False
        self.node_folder = node_folder
        self.node_files = []

        #labels folder
        self.labels_enabled = False
        self.label_folder = label_folder
        self.label_files = []

        #variable to keep track of the number of frames
        self.num_frames = 0

        #load the new dataset
        self.load_new_dataset(dataset_path)

        return

    def load_new_dataset(self,dataset_path:str):

        self.dataset_path = dataset_path
        self.import_dataset_files()
        self.determine_num_frames()

    def import_dataset_files(self):

        self.import_node_data()
        self.import_label_data()
            
    def determine_num_frames(self):

        self.num_frames = 0

        if self.nodes_enabled:
            self.set_num_frames(len(self.node_files))
        if self.labels_enabled:
            self.set_num_frames(len(self.label_files))     
        return
    
    def set_num_frames(self,num_files:int):
        """Update the number of frames available in the dataset

        Args:
            num_files (int): The number of files available for a given sensor
        """
        if self.num_frames > 0:
            self.num_frames = min(self.num_frames,num_files)
        else:
            self.num_frames = num_files
    
    ####################################################################
    #handling node data
    ####################################################################   
    def import_node_data(self):

        path = os.path.join(self.dataset_path,self.node_folder)

        if os.path.isdir(path):
            self.node_files = sorted(os.listdir(path))
            self.nodes_enabled = True
            print("found {} node samples".format(len(self.node_files)))
        else:
            print("did not find node samples")

        return
      
    def get_node_data(self,idx:int)->np.ndarray:
        """Get save node samples

        Args:
            idx (int): The index of the node data sample

        Returns:
            np.ndarray: NxM array of N nodes with M properties per Node
        """

        assert self.nodes_enabled, "No nodes dataset loaded"

        path = os.path.join(
            self.dataset_path,
            self.node_folder,
            self.node_files[idx])
        
        nodes = np.load(path)
                
        return nodes
    
    ####################################################################
    #handling label data
    ####################################################################   
    def import_label_data(self):

        path = os.path.join(self.dataset_path,self.label_folder)

        if os.path.isdir(path):
            self.label_files = sorted(os.listdir(path))
            self.labels_enabled = True
            print("found {} label samples".format(len(self.label_files)))
        else:
            print("did not find label samples")

        return
      
    def get_label_data(self,idx:int)->np.ndarray:
        """Get saved label samples

        Args:
            idx (int): The index of the label data sample

        Returns:
            np.ndarray: N-element array of N nodes where 1 indicates the node
                valid and 0 indicates the node is invalid
        """

        assert self.labels_enabled, "No labels dataset loaded"

        path = os.path.join(
            self.dataset_path,
            self.label_folder,
            self.label_files[idx])
        
        nodes = np.load(path)
                
        return nodes