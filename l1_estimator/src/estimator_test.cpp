#include "estimator_test/estimator_test_tools.h"
#include <assert.h>
#include "estimator_test/plot_tools_for_estimator_test.h"


double magnitude = 2;

void plot_all_data();

void plot_position_data();

void plot_linear_velocity_data();

void plot_quaternion_data();

void plot_angular_velocity_data();

void plot_sigma_data();

void plot_theta_data();

int main(int argc, char**argv)
{
    ros::init(argc, argv, "Test node");
    
    // Declare ROS NodeHandle to get yaml directory.
    ros::NodeHandle nh;

    param_setup(nh);

    rpm.setZero();

    mat31_t u_comp, mu_comp;
    u_comp.setZero();
    mu_comp.setZero();



    assert(simulation_time.capacity() == N);

    for(size_t i = 0; i < N - 1; i++)
    {
        sigma_ext << 5*cos(3*simulation_time[i]),
        2*cos(1*simulation_time[i]),
        3*sin(2*simulation_time[i]);

        theta_ext << magnitude*sin(1*simulation_time[i]),
        magnitude*cos(1*simulation_time[i]),
        -magnitude*cos(1*simulation_time[i]);


        play_simulation_model(rpm, sigma_ext, theta_ext, simulation_time[i+1]);

        // Get state from the simulator model.
        simulation_model_ptr->get_state(p_state, v_state, 
        q_state, w_state);

        // Set control input, measured state, disturbance and simulation time
        reference_model_ptr->set_input_state_disturbance_time(u_comp, mu_comp,
        p_state_prev, v_state_prev, q_state_prev, w_state_prev,
        sigma_est_noisy, theta_est_noisy, simulation_time[i+1]);

        // Integrate the reference model (Prediction step)
        reference_model_ptr->prediction();

        // Get state from the reference model (Prediction)
        reference_model_ptr->get_state_from_ref_model(p_ref, v_ref, q_ref, w_ref);

        p_state_prev = p_state;
        v_state_prev = v_state;
        q_state_prev = q_state;
        w_state_prev = w_state;

        // Set disturbance estimator
        disturbance_est_ptr->set_state_time(p_state, p_ref, 
        v_state, v_ref, q_state, q_ref, 
        w_state, w_ref, simulation_time[i+1]);

        disturbance_est_ptr->solve();

        disturbance_est_ptr->get_est_raw(sigma_est_noisy, theta_est_noisy);

        disturbance_est_ptr->get_est_filtered(sigma_est_lpf, theta_est_lpf);


        demux_simulation_state(p_state, v_state, q_state, w_state);
        demux_reference_state(p_ref, v_ref, q_ref, w_ref);
        demux_disturbance_ext(sigma_ext, theta_ext);
        demux_disturbance_est_noisy(sigma_est_noisy, theta_est_noisy);
        demux_disturbance_est_filtered(sigma_est_lpf, theta_est_lpf);

        if(i <= 100)
        {
            print_state(simulation_time[i+1], p_state, v_state, q_state, w_state,
            p_ref, v_ref, q_ref, w_ref, theta_ext, sigma_ext, theta_est_lpf,
            sigma_est_lpf, theta_est_noisy, sigma_est_noisy);
        }

    }

    cout << "At final time step" << endl;
    print_state(simulation_time[N-1], p_state, v_state, q_state, w_state,
    p_ref, v_ref, q_ref, w_ref, theta_ext, sigma_ext, theta_est_lpf,
    sigma_est_lpf, theta_est_noisy, sigma_est_noisy);

    keywords_setup(line_width, font_size);

    // plt::figure_size(3500,2000);
    // plot_quaternion_data();

    plt::figure_size(3500,2000);
    plot_theta_data();

    plt::figure_size(3500,2000);
    plot_sigma_data();

    plt::show();


    delete simulation_model_ptr;
    delete reference_model_ptr;
    delete disturbance_est_ptr;

    return EXIT_SUCCESS;
}

void plot_all_data()
{
    plot_position_data();

    plot_linear_velocity_data();

    plot_quaternion_data();

    plot_angular_velocity_data();

    plot_sigma_data();

    plot_theta_data();

    plt::show();
}

void plot_position_data()
{

    ticks_setup(Tf, 20, 0, 5, 5);    
    y_label = "x";
    data1_label = "x ref";
    data2_label = "x state";

    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,x_ref, x_state);
    plt::show();

    ticks_setup(Tf, 20, 0, 5, 5);

    y_label = "y";
    data1_label = "y ref";
    data2_label = "y state";

    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,y_ref, y_state);
    plt::show();


    ticks_setup(Tf, 20, 0, 5, 5);    
    y_label = "z";
    data1_label = "z ref";
    data2_label = "z state";

    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,z_ref, z_state);
    plt::show();

}

void plot_linear_velocity_data()
{
    ticks_setup(Tf, 20, 0, 5, 5);    
    y_label = "vx";
    data1_label = "vx ref";
    data2_label = "vx state";

    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,vx_ref, vx_state);
    plt::show();

    ticks_setup(Tf, 20, 0, 5, 5);

    y_label = "vy";
    data1_label = "vy ref";
    data2_label = "vy state";

    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,vy_ref, vy_state);
    plt::show();


    ticks_setup(Tf, 20, 0, 5, 5);    
    y_label = "vz";
    data1_label = "vz ref";
    data2_label = "vz state";

    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,vz_ref, vz_state);
    plt::show();

}

void plot_quaternion_data()
{
    ticks_setup(Tf, 1, -1, 5, 5);
    
    plt::subplot(1,4,1);
    y_label = "$q_{w}$";
    data1_label = "$q_{w,ref}$";
    data2_label = "$q_{w,state}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,qw_ref , qw_state);

    ticks_setup(Tf, 1, -1, 5, 5);
    plt::subplot(1,4,2);    
    y_label = "$q_{x}$";
    data1_label = "$q_{x,ref}$";
    data2_label = "$q_{x,state}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,qx_ref , qx_state);


    ticks_setup(Tf, 1, -1, 5, 5);
    plt::subplot(1,4,3);
    y_label = "$q_{y}$";
    data1_label = "$q_{y,ref}$";
    data2_label = "$q_{y,state}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,qy_ref , qy_state);


    ticks_setup(Tf, 1, -1, 5, 5);
    plt::subplot(1,4,4);
    y_label = "$q_{z}$";
    data1_label = "$q_{z,ref}$";
    data2_label = "$q_{z,state}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,qz_ref , qz_state);
}

void plot_angular_velocity_data()
{
    ticks_setup(Tf, 20, 0, 5, 5);    
    y_label = "wx";
    data1_label = "wx ref";
    data2_label = "wx state";

    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,wx_ref, wx_state);
    plt::show();

    ticks_setup(Tf, 20, 0, 5, 5);

    y_label = "wy";
    data1_label = "wy ref";
    data2_label = "wy state";

    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,wy_ref, wy_state);
    plt::show();


    ticks_setup(Tf, 20, 0, 5, 5);    
    y_label = "wz";
    data1_label = "wz ref";
    data2_label = "wz state";

    plot_data(simulation_time, y_label, 
    data1_label, data2_label ,wz_ref, wz_state);
}

void plot_sigma_data()
{

    plt::subplot(3,1,1);
    ticks_setup(Tf, 4, -4, 5, 5);    
    y_label = "$σ_{x}$";
    data1_label = "$σ_{x, est}$";
    data2_label = "$σ_{x, est, lpf}$";
    data3_label = "$σ_{x, ext}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label, data3_label,
    sigma_est_x, sigma_est_lpf_x, sigma_ext_x);
    plt::title("$σ_{x}$ - t", label_keywords);


    plt::subplot(3,1,2);
    ticks_setup(Tf, 4, -4, 5, 5);    
    y_label = "$σ_{y}$";
    data1_label = "$σ_{y, est}$";
    data2_label = "$σ_{y, est, lpf}$";
    data3_label = "$σ_{y, ext}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label, data3_label,
    sigma_est_y, sigma_est_lpf_y, sigma_ext_y);
    plt::title("$σ_{y}$ - t", label_keywords);

    plt::subplot(3,1,3);
    ticks_setup(Tf, 4, -4, 5, 5);    
    y_label = "$σ_{z}$";
    data1_label = "$σ_{z, est}$";
    data2_label = "$σ_{z, est, lpf}$";
    data3_label = "$σ_{z, ext}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label, data3_label,
    sigma_est_z, sigma_est_lpf_z, sigma_ext_z);
    plt::title("$σ_{z}$ - t", label_keywords);

}

void plot_theta_data()
{
    plt::subplot(3,1,1);
    ticks_setup(Tf, magnitude, -magnitude, 5, 3);    
    y_label = "$θ_{x}$";
    data1_label = "$θ_{x, est}$";
    data2_label = "$θ_{x, est, lpf}$";
    data3_label = "$θ_{x, ext}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label, data3_label,
    theta_est_x, theta_est_lpf_x, theta_ext_x);
    plt::title("$θ_{x}$ - t", label_keywords);

    plt::subplot(3,1,2);
    ticks_setup(Tf, magnitude, -magnitude, 5, 3);    
    y_label = "$θ_{y}$";
    data1_label = "$θ_{y, est}$";
    data2_label = "$θ_{y, est, lpf}$";
    data3_label = "$θ_{y, ext}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label, data3_label 
    ,theta_est_y, theta_est_lpf_y, theta_ext_y);
    plt::title("$θ_{y}$ - t", label_keywords);

    plt::subplot(3,1,3);
    ticks_setup(Tf, magnitude, -magnitude, 5, 3);    
    y_label = "$θ_{z}$";
    data1_label = "$θ_{z, est}$";
    data2_label = "$θ_{z, est, lpf}$";
    data3_label = "$θ_{z, ext}$";
    plot_data(simulation_time, y_label, 
    data1_label, data2_label, data3_label,
    theta_est_z, theta_est_lpf_z, theta_ext_z);
    plt::title("$θ_{z}$ - t", label_keywords);

}
