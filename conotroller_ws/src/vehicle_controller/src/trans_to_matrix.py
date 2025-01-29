import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np

class OccupancyGridListener:
    def __init__(self):
        rospy.init_node('occupancy_grid_listener', anonymous=True)
        self.subscriber = rospy.Subscriber('/projected_map', OccupancyGrid, self.grid_callback)
        rospy.loginfo("OccupancyGrid listener initialized and waiting for messages...")

    def grid_callback(self, msg):
        rospy.loginfo("Received OccupancyGrid message")
        
        # Extract grid map info
        resolution = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        origin = msg.info.origin
        rospy.loginfo(f"Map resolution: {resolution}, width: {width}, height: {height}")
        rospy.loginfo(f"Map origin: {origin.position.x}, {origin.position.y}, {origin.position.z}")

        # Convert 1D data array to 2D NumPy array
        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        print(grid_data.shape)

        # Example: Print a subset of the grid
        rospy.loginfo(f"Grid data: \n{grid_data}")

        # Save the grid to a file if needed
        self.save_grid_to_file(grid_data, "grid_map.txt")

    def save_grid_to_file(self, grid_data, filename):
        np.savetxt(filename, grid_data, fmt="%d")
        rospy.loginfo(f"Grid saved to {filename}")


if __name__ == "__main__":
    try:
        listener = OccupancyGridListener()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("OccupancyGrid listener shut down.")
    except KeyboardInterrupt:
        rospy.loginfo("Exiting OccupancyGrid listener.")
