#include "l1_estimator/ros_lpf_node.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lpf_test_node");

    NodeHandle nh;

    double tau;

    nh.getParam("tau", tau);

    RosLpf ros_lpf_obj(nh, tau);

    ros::Rate loop_rate(100);

    while(ros::ok())
    {
        ros::spinOnce();
        ros_lpf_obj.filtered_signal_publish();
        loop_rate.sleep();
    }

    return EXIT_SUCCESS;

}