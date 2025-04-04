#include "ros_lpf_node.hpp"

RosLpf::RosLpf(const NodeHandle &nh, const double &tau)
:nh_(nh), lpf_obj(Lpf(tau)), 
signal_filtered_(signal_filtered_.setZero()),
time_curr(0), time_offset(0),
init_time(false)
{
    ros_setup();
}

void RosLpf::ros_setup()
{
    signal_subscriber_ = nh_.subscribe("/input_signal",
    1,
    &RosLpf::callback,
    this);
    lpf_publisher_ = nh_.advertise<Lpf_test>("/filtered_signal", 1);

}

void RosLpf::callback(const Lpf_testConstPtr &signal_msg)
{
    mat31_t v_in;
    switch (init_time)
    {
        case true:
            time_curr =  
            signal_msg->stamp.sec
            + (signal_msg->stamp.nsec)*1e-9
            - time_offset;
            
            for(size_t i = 0; i < v_in.size(); i++)
            {
                v_in(i) = signal_msg->v[i];
            }

            lpf_obj.set_input_and_time(v_in, time_curr);

            // Integrate.
            lpf_obj.solve();

            // Integrate and get the filtered signal
            lpf_obj.get_filtered_vector(signal_filtered_);
            break;
        
        case false:
            // Before filtering the signal,
            // get the time offset.
            time_offset = signal_msg->stamp.sec
            + (signal_msg->stamp.nsec)*1e-9;
            init_time = true;
            break;
    }

}

void RosLpf::filtered_signal_publish()
{
    Lpf_test filtered_msg;
    filtered_msg.header.stamp = ros::Time::now();
    filtered_msg.stamp = ros::Time::now();
    
    for(size_t i = 0; i < 3; i++)
        filtered_msg.v[i] = signal_filtered_(i);

    lpf_publisher_.publish(filtered_msg);
}