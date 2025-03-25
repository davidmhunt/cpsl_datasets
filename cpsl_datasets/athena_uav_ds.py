import os
import numpy as np
import matplotlib.image as img #not needed for ROS
from geometries.pose.pose import Pose
from geometries.pose.orientation import Orientation
from geometries.pose.position import Position
from geometries.coordinate_systems import coordinate_system_conversions
from geometries.transforms.transformation import Transformation

class AthenaUAVDS:

    def __init__(self,
                 dataset_path,
                 depth_sensor_file="depth.txt",
                 state_dict_file="state.txt",
                 ) -> None:
        """Initialize a new Athena UAV Dataset class

        Args:
            dataset_path (_type_): the path to the folder containing the depth.txt and state.txt file
            depth_sensor_file (str, optional): the name of the file containing the depth data. Defaults to "depth.txt".
            state_dict_file (str, optional): the name of the file containing the state data. Defaults to "state.txt".
        """
      
        #depth data
        self.depth_data_file = depth_sensor_file
        self.depth_data = np.empty(shape=(0,7,7))
        self.depth_time_stamps = np.empty(shape=(0)) #in seconds
        self.depth_sampling_period = 0.0 #in seconds
        self.valid_data_idxs = np.empty(shape=0)

        #state data [x,y,z,roll,pitch,yaw]
        self.state_data_file = state_dict_file
        self.state_data = np.empty(shape=(0,6))
        self.state_time_stamps = np.empty(shape=0) #in seconds
        self.state_sampling_period = 0.0 #in seconds

        #variable to keep track of the number of frames
        self.num_frames = 0

        #initialize the transformations for processing the data
        self.front_transform:Transformation = None
        self.left_transform:Transformation = None
        self.right_transform:Transformation = None
        self.init_transforms()

        #load the new dataset
        self.load_new_dataset(dataset_path)

        return

    def load_new_dataset(self,dataset_path:str):
        """load a new dataset

        Args:
            dataset_path (str): the full path to the dataset
        """
        self.dataset_path = dataset_path
        self.import_dataset_files()
        self.determine_num_frames()

    def import_dataset_files(self):
        """Import all dataset files
        """
        self.import_state_data()
        self.import_depth_data()

        #get the list of valid depth data
        self.set_valid_depth_data_idxs()
            
    def determine_num_frames(self):
        """Initialize the total number of frames in the datset
        """
        self.num_frames = self.valid_data_idxs.shape[0]

        return
    
    def init_transforms(self):
        """Initialize the transformation to apply to UAV data
        """
        #initialize front transformation
        front_rotation = Orientation.from_euler(
            roll=0.0,pitch=0.0,yaw=0.0,degrees=True
        )
        self.front_transform = Transformation(
            translation=np.array([-0.07,0.0,0.0]), #[x,y,z]
            rotation=front_rotation._orientation
        )

        #initialize right transformation
        right_rotation = Orientation.from_euler(
            roll=0.0,pitch=0.0,yaw=-90.0,degrees=True
        )
        self.right_transform = Transformation(
            translation=np.array([0.0,-0.07,0.0]), #[x,y,z]
            rotation=right_rotation._orientation
        )

        #initialize left transformation
        left_rotation = Orientation.from_euler(
            roll=0.0,pitch=0.0,yaw=90.0,degrees=True
        )
        self.left_transform = Transformation(
            translation=np.array([0.0,0.07,0.0]), #[x,y,z]
            rotation=left_rotation._orientation
        )
    ####################################################################
    #handling state data
    ####################################################################   
    def import_state_data(self):
        """Import state data from the given dataset
        """
        path = os.path.join(self.dataset_path,self.state_data_file)

        if os.path.isfile(path):

            timesteps = []
            values = []
            
            with open(path, 'r') as file:
                for line in file:
                    if line.strip() == "":
                        continue
                    parts = line.split(':')
                    timestep = 1e-3 * float(parts[0].strip())
                    value_list = 1e-3 * np.fromstring(parts[1].strip().strip('[]'), sep=' ', dtype=float)
                    
                    timesteps.append(timestep)
                    values.append(value_list)
            
            self.state_time_stamps = np.array(timesteps)
            self.state_data = np.vstack(values)

            #compute the average period
            diffs = self.state_time_stamps[1:] - self.state_time_stamps[:-1]
            self.state_sampling_period = np.average(diffs)

            print("loaded {} state dict samples".format(self.state_data.shape[0]))
        else:
            print("did not find state dict path")

        return
    
    def get_state_data_raw(self,idx:int)->np.ndarray:
        """Get radar detections or ADC data cube for a specific index in the dataset

        Args:
            idx (int): The index of the state sample to get

        Returns:
            np.ndarray: An Nx6 of radar detections with (x,y,z,roll,pitch,yaw) vals
        """

        assert self.state_data.shape[0] > 0, "No radar dataset loaded"
                    
        return self.state_data[idx]
    
    def get_state_data_raw_at_stamp(self,stamp_s:float)->np.ndarray:
        """Get the interpolated state for a given time stamp

        Args:
            stamp_s (float): desired time in seconds to get the state data from

        Raises:
            ValueError: if the Target time cannot be found in the dataset

        Returns:
            np.ndarray: [x,y,z,roll,pitch,yaw] values interpolated for the given time stamp
        """
        idx = np.searchsorted(self.state_time_stamps,stamp_s)

        if idx ==0 or idx == self.state_time_stamps.shape[0]:
            raise ValueError ("Target time is out of bounds")
        
        #get time estimates
        t1 = self.state_time_stamps[idx] - 1
        t2 = self.state_time_stamps[idx]

        #get state values
        v1:np.ndarray = self.state_data[idx-1,:]
        v2:np.ndarray = self.state_data[idx,:]

        interpolated_values = v1 + (v1 - v2) * (stamp_s - t1) / (t2-t1)

        # Handle angular wrap-around for roll, pitch, yaw (last 3 values)
        for i in range(3, 6):  # Indices for roll, pitch, yaw
            if abs(v2[i] - v1[i]) > np.pi:
                # Adjust for wrap-around
                if v2[i] > v1[i]:
                    v1[i] += 2 * np.pi
                else:
                    v2[i] += 2 * np.pi

            # Perform interpolation in the adjusted range
            interpolated_values[i] = v1[i] + (v2[i] - v1[i]) * (stamp_s - t1) / (t2 - t1)

            # Wrap back to [-pi, pi]
            interpolated_values[i] = (interpolated_values[i] + np.pi) % (2 * np.pi) - np.pi

        return interpolated_values
    
    def get_state_data_at_stamp(self,stamp_s:float) -> Pose:
        """Get the pose data (interpolate time if needed) at a specific time stamp

        Args:
            stamp_s (float): the time to obtain the pose at

        Returns:
            Pose: a Pose object representing the pose of the UAV at the given time stamp
        """
        #get the raw state data [x,y,z,roll,pitch,yaw]
        state = self.get_state_data_raw_at_stamp(stamp_s)

        pose = Pose(
            position=Position(
                x=state[0],
                y=state[1],
                z=state[2]
            ),
            orientation=Orientation.from_euler(
                roll=state[3],
                pitch=state[4],
                yaw=state[5],
                degrees=False
            )
        )

        return pose

    
    ####################################################################
    #handling depth data
    ####################################################################   
    def import_depth_data(self):
        """Import depth data from the dataset
        """
        path = os.path.join(self.dataset_path,self.depth_data_file)

        if os.path.isfile(path):

            #dataset variables
            timesteps = []
            dataset_values = []

            #sample variables
            timestep = 0.0
            sample_values = []
            
            with open(path, 'r') as file:
                for line in file:
                    if line.strip("\n") == "": #end of a dataset sample
                        timesteps.append(timestep)
                        dataset_values.append(np.vstack(sample_values))

                        #reset sample values array
                        sample_values = []
                    elif ":" in line: #starting a new dataset
                        parts = line.split(':')
                        timestep = 1e-3 * float(parts[0].strip())
                        value_list = 1e-3 * np.fromstring(parts[1].strip().strip('[]'), sep=' ', dtype=float)
                        sample_values.append(value_list)
                    else: #another depth entry for a dataset sample
                        value_list = 1e-3 * np.fromstring(line.strip().strip('[]'), sep=' ', dtype=float)
                        sample_values.append(value_list)
                        
            
            self.depth_time_stamps = np.array(timesteps)
            self.depth_data = np.stack(dataset_values,axis=0)

            diffs = self.depth_time_stamps[1:] - self.depth_time_stamps[:-1]
            self.depth_sampling_period = np.average(diffs)

            print("loaded {} depth samples".format(self.state_data.shape[0])) 
        else:
            print("did not find depth data")

        return
    
    def set_valid_depth_data_idxs(self):
        """Initialize an array of valid depth measurement indicies. 
        Valid measurements have a time stamp that can be interpolated 
        from the given state data
        """

        assert self.depth_data.shape[0] > 0 and self.state_data.shape[0] > 0

        self.valid_data_idxs = np.where(
            (self.depth_time_stamps > self.state_time_stamps[0]) & 
            (self.depth_time_stamps < self.state_time_stamps[-1])
        )[0]

        print("using {} depth samples with valid time steps".format(
            self.valid_data_idxs.shape[0]
        ))

    def _grid_to_pc(self,grid:np.ndarray)->np.ndarray:
        """convert an 8-element grid of depth samples from (-22.5,22.5) degrees
        to a cartesian point cloud

        Args:
            grid (np.ndarray): 8-element grid of depth samples

        Returns:
            np.ndarray: cartesian point cloud in [x,y,z] coordinates. Invalid detections
                are automatically filtered out
        """

        #initialize the point cloud in spherical coordinates (r,theta, phi)
        pts_spherical = np.zeros(
            shape=(8,3),
            dtype=float
        )
        #fill in the ranges
        pts_spherical[:,0] = grid

        #compute yaw angles
        pts_spherical[:,1] = np.linspace(
            start=np.deg2rad(-45/2),
            stop=np.deg2rad(45/2),
            num=8)
        
        pts_spherical[:,2] = np.pi/2
        
        #remove points that have a negative distance
        pts_spherical = pts_spherical[pts_spherical[:,0]>0]

        #convert the points to cartesian coordinates
        if pts_spherical.shape[0] > 0:
            pts_cartesian = \
                coordinate_system_conversions.spherical_to_cartesian(
                    pts_spherical
                )
            
            return pts_cartesian
        
        else:
            return np.empty(shape=(0,3))
        
    def depth_sample_to_pc(self,depth_grid:np.ndarray)->np.ndarray:
        """Convert a 8x8 np array of depth sensor readings form a UAV to 
        a 3D point cloud

        Args:
            depth_grid (np.ndarray): _description_

        Returns:
            np.ndarray: Nx3 array of points (x,y,z)
        """

        #get the left points
        left_pts = self._grid_to_pc(depth_grid[0,:])
        left_pts = self.left_transform.apply_transformation(left_pts)

        #get the front points
        front_pts = self._grid_to_pc(depth_grid[4,:])
        front_pts = self.front_transform.apply_transformation(front_pts)

        #get the right points
        right_pts = self._grid_to_pc(depth_grid[7,:])
        right_pts = self.right_transform.apply_transformation(right_pts)

        pts = np.vstack((left_pts,front_pts,right_pts))

        return pts
    


    ####################################################################
    #accessing data from the dataset
    ####################################################################  
    def get_data_at_idx(self,valid_idx:int)->tuple:
        """Get the pose and point cloud from a specific index in the dataset

        Args:
            valid_idx (int): a valid index from the dataset

        Returns:
            tuple: (pose,point cloud) where pose is a Pose class, and point cloud
                is a np.ndarray of shape Nx3 points corresponding to (x,y,z)
        """
        depth_sample_idx = self.valid_data_idxs[valid_idx]

        #get the time stamp
        stamp = self.depth_time_stamps[depth_sample_idx]

        #get the pose
        pose = self.get_state_data_at_stamp(stamp_s=stamp)

        #get the point cloud
        depth_grid = self.depth_data[depth_sample_idx,:,:]
        pts = self.depth_sample_to_pc(depth_grid)

        return pose,pts