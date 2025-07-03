import os
import numpy as np
import matplotlib.image as img #not needed for ROS

class CpslDS:

    def __init__(self,
                 dataset_path,
                 radar_folder="radar_0",
                 lidar_folder="lidar",
                 camera_folder="camera",
                 imu_orientation_folder="imu_data",
                 imu_full_folder="imu_full",
                 vehicle_vel_folder="vehicle_vel",
                 vehicle_odom_folder="vehicle_odom"

                 ) -> None:
        
      
        #radar data
        self.radar_enabled = False
        self.radar_folder = radar_folder
        self.radar_files = []

        #lidar data
        self.lidar_enabled = False
        self.lidar_folder = lidar_folder
        self.lidar_files = []

        #camera data
        self.camera_enabled = False
        self.camera_folder = camera_folder
        self.camera_files = []

        #imu data - orientation only
        self.imu_orientation_folder = imu_orientation_folder
        self.imu_orientation_files = []
        self.imu_orientation_enabled = False

        #imu data - full sensor data
        self.imu_full_folder = imu_full_folder
        self.imu_full_files = []
        self.imu_full_enabled = False

        #vehicle velocity data
        self.vehicle_vel_folder = vehicle_vel_folder
        self.vehicle_vel_files = []
        self.vehicle_vel_enabled = False

        #vehicle odometry data
        self.vehicle_odom_folder = vehicle_odom_folder
        self.vehicle_odom_files = []
        self.vehicle_odom_enabled = False

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

        self.import_radar_data()
        self.import_lidar_data()
        self.import_camera_data()
        self.import_imu_orientation_data()
        self.import_imu_full_data()  
        self.import_vehicle_vel_data()
        self.import_vehicle_odom_data()
            
    def determine_num_frames(self):

        self.num_frames = 0

        if self.radar_enabled:
            self.set_num_frames(len(self.radar_files))
        if self.lidar_enabled:
            self.set_num_frames(len(self.lidar_files))
        if self.camera_enabled:
            self.set_num_frames(len(self.camera_files))
        if self.imu_full_enabled:
            self.set_num_frames(len(self.imu_full_files))
        if self.imu_orientation_enabled:
            self.set_num_frames(len(self.imu_orientation_files))
        if self.vehicle_vel_enabled:
            self.set_num_frames(len(self.vehicle_vel_files))
        
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
    #handling radar data
    ####################################################################   
    def import_radar_data(self):

        path = os.path.join(self.dataset_path,self.radar_folder)

        if os.path.isdir(path):
            self.radar_enabled = True
            self.radar_files = sorted(os.listdir(path))
            print("found {} radar samples".format(len(self.radar_files)))
        else:
            print("did not find radar samples")

        return
    
    def get_radar_data(self,idx:int)->np.ndarray:
        """Get radar detections or ADC data cube for a specific index in the dataset

        Args:
            idx (int): The index of the radar data detection

        Returns:
            np.ndarray: An Nx4 of radar detections with (x,y,z,vel) vals or 
                        (rx_channels) x (samples) x (chirps) ADC cube for a given frame
        """

        assert self.radar_enabled, "No radar dataset loaded"

        path = os.path.join(
            self.dataset_path,
            self.radar_folder,
            self.radar_files[idx])
        
        points = np.load(path)
                
        return points
    
    ####################################################################
    #handling lidar data
    ####################################################################   
    def import_lidar_data(self):

        path = os.path.join(self.dataset_path,self.lidar_folder)

        if os.path.isdir(path):
            self.lidar_enabled = True
            self.lidar_files = sorted(os.listdir(path))
            print("found {} lidar samples".format(len(self.lidar_files)))
        else:
            print("did not find lidar samples")

        return
    
    def get_lidar_point_cloud(self,idx)->np.ndarray:
        path = os.path.join(
            self.dataset_path,
            self.lidar_folder,
            self.lidar_files[idx]
        )
        """Get a lidar pointcloud from the desired frame,
        filters out ground and higher detections points

        Returns:
            np.ndarray: a Nx2 array of lidar detections
        """
        assert self.lidar_enabled, "No lidar dataset loaded"
        points = np.load(path)

        valid_points = points[:,2] > -0.2 #filter out ground
        valid_points = valid_points & (points[:,2] < 0.1) #higher elevation points

        points = points[valid_points,:2]

        return points
    
    def get_lidar_point_cloud_raw(self,idx)->np.ndarray:
        path = os.path.join(
            self.dataset_path,
            self.lidar_folder,
            self.lidar_files[idx]
        )
        """Get a lidar pointcloud from the desired frame,
        without filtering anything out

        Returns:
            np.ndarray: a Nx3 array of lidar detections
        """
        assert self.lidar_enabled, "No lidar dataset loaded"
        points = np.load(path)

        return points
        
    ####################################################################
    #handling camera data
    ####################################################################   
    def import_camera_data(self):

        path = os.path.join(self.dataset_path,self.camera_folder)

        if os.path.isdir(path):
            self.camera_enabled = True
            self.camera_files = sorted(os.listdir(path))
            print("found {} camera samples".format(len(self.camera_files)))
        else:
            print("did not find camera samples")

        return

    def get_camera_frame(self,idx:int)->np.ndarray:
        """Get a camera frame from the dataset

        Args:
            idx (int): the index in the dataset to get the camera
                data from

        Returns:
            np.ndarray: the camera data with rgb channels
        """
        assert self.camera_enabled, "No camera dataset loaded"

        path = os.path.join(
            self.dataset_path,
            self.camera_folder,
            self.camera_files[idx])
        image = img.imread(path)

        #return while also flipping red and blue channel
        # return image[:,:,::-1]
        return image
    
    ####################################################################
    #handling imu data (orientation only)
    ####################################################################   
    def import_imu_orientation_data(self):

        path = os.path.join(self.dataset_path,self.imu_orientation_folder)

        if os.path.isdir(path):
            self.imu_orientation_enabled = True
            self.imu_orientation_files = sorted(os.listdir(path))
            print("found {} imu (orientation only) samples".format(len(self.imu_orientation_files)))
        else:
            print("did not find imu (orientation) samples")

        return
    
    def get_imu_orientation_rad(self,idx:int):
        """Get the raw imu heading from the dataset at a given frame index

        Args:
            idx (int): the frame index to get the imu heading for

        Returns:
            _type_: the raw heading read from the IMU expressed in the range
                [-pi,pi]
        """
        assert self.imu_orientation_enabled, "No IMU (orientation) dataset loaded"

        path = os.path.join(
            self.dataset_path,
            self.imu_orientation_folder,
            self.imu_orientation_files[idx]
        )

        data = np.load(path)
        w = data[0]
        x = data[1]
        y = data[2]
        z = data[3]

        heading = np.arctan2(
            2 * (w * z + x * y), 1 - 2 * (y * y + z * z)
        )
        return heading
    
    ####################################################################
    #handling imu (full sensor) data
    ####################################################################   
    def import_imu_full_data(self):

        path = os.path.join(self.dataset_path,self.imu_full_folder)

        if os.path.isdir(path):
            self.imu_full_enabled = True
            self.imu_full_files = sorted(os.listdir(path))
            print("found {}imu (full data) samples".format(len(self.imu_full_files)))
        else:
            print("did not find imu (full data) samples")

        return
    
    def get_imu_full_data(self,idx=0):
        """_summary_

        Args:
            idx (int, optional): _description_. Defaults to 0.

        Returns:
            np.ndarray: [time,w_x,w_y,w_z,acc_x,acc_y,acc_z]
        """
        assert self.imu_full_enabled, "No IMU Full dataset loaded"

        #load the data sample
        path = os.path.join(
            self.dataset_path,
            self.imu_full_folder,
            self.imu_full_files[idx])

        return np.load(path)
    
    ####################################################################
    #handling vehicle velocity data
    ####################################################################
    def import_vehicle_vel_data(self):

        path = os.path.join(self.dataset_path,self.vehicle_vel_folder)

        if os.path.isdir(path):
            self.vehicle_vel_enabled = True
            self.vehicle_vel_files = sorted(os.listdir(path))
            print("found {} vehicle velocity samples".format(len(self.vehicle_vel_files)))
        else:
            print("did not find vehicle velocity samples")

        return
    
    def get_vehicle_vel_data(self,idx=0):

        assert self.vehicle_vel_files, "No Vehicle velocity dataset loaded"

        #load the data sample
        path = os.path.join(
            self.dataset_path,
            self.vehicle_vel_folder,
            self.vehicle_vel_files[idx])

        return np.load(path)
    
    ####################################################################
    #handling vehicle odometry data
    ####################################################################
    def import_vehicle_odom_data(self):

        path = os.path.join(self.dataset_path,self.vehicle_odom_folder)

        if os.path.isdir(path):
            self.vehicle_odom_enabled = True
            self.vehicle_odom_files = sorted(os.listdir(path))
            print("found {} vehicle odometry samples".format(len(self.vehicle_odom_files)))
        else:
            print("did not find vehicle odometry samples")

        return
    
    def get_vehicle_odom_data(self,idx=0):
        """Returns  [time,x,y,z,quat_w,quat_x,quat_y,quat_z,vx,vy,vz,wx,wy,wz]

        Args:
            idx (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """

        assert self.vehicle_odom_enabled, "No Vehicle odometry dataset loaded"

        #load the data sample
        path = os.path.join(
            self.dataset_path,
            self.vehicle_odom_folder,
            self.vehicle_odom_files[idx])

        return np.load(path)