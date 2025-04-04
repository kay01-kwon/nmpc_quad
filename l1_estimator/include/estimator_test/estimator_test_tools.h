#ifndef ESTIMATOR_TEST_TOOLS_H_
#define ESTIMATOR_TEST_TOOLS_H_

#include <ros/ros.h>
#include <numeric>
#include "l1_estimator/l1_estimator.hpp"
#include "yaml_converter/read_config.hpp"
#include "simulation_model/simulation_model.hpp"
#include "variable_def.h"

void param_setup(const ros::NodeHandle& nh);

void print_parameter_setup(const inertial_param_t& inertial_param, 
    const aero_coeff_t& aero_param,
    const mat31_t& bound_theta, const double& epsilon_theta,
    const mat31_t& bound_sigma, const double& epsilon_sigma,
    const mat33_t& Gamma_sigma, const mat33_t& Gamma_theta,
    const double& tau_sigma, const double& tau_theta);

void play_simulation_model(const mat41_t& rpm_, 
const mat31_t& sigma_ext_, const mat31_t& theta_ext_,
const double& simulation_time_);

void variable_capacity_reserve(const int& N);

void demux_simulation_state(const mat31_t& position,
const mat31_t& linear_velocity,
const quat_t& quaternion,
const mat31_t& angular_velocity);

void demux_reference_state(const mat31_t& position,
const mat31_t& linear_velocity,
const quat_t& quaternion,
const mat31_t& angular_velocity);

void demux_disturbance_ext(const mat31_t& sigma_,
const mat31_t& theta_);

void demux_disturbance_est_noisy(const mat31_t& sigma_,
const mat31_t& theta_);

void demux_disturbance_est_filtered(const mat31_t& sigma_,
const mat31_t& theta_);

void demux_vec3(const mat31_t& v, vector<double>&x, vector<double>&y, vector<double>&z);

void demux_quat(const quat_t& q, vector<double>&qw, vector<double>&qx, vector<double>&qy, vector<double>&qz);

void print_state(const double& time_, 
const mat31_t& p_state_, const mat31_t& v_state_,
const quat_t &q_state_, const mat31_t &w_state_,
const mat31_t &p_ref_, const mat31_t &v_ref_, 
const quat_t &q_ref_, const mat31_t &w_ref_,
const mat31_t &theta_ext_, const mat31_t &sigma_ext_,
const mat31_t &theta_est_lpf_, const mat31_t &sigma_est_lpf_,
const mat31_t &theta_est_noisy_, const mat31_t &sigma_est_noisy_);

void initialilze_temporary_data_to_store_min_max();

void find_max(const double& data, double& dest);

void find_min(const double& data, double& dest);



/**
 * @brief From ros node handle get parameters
 * such as inertial param, reference model param, 
 * disturbance param and so on
 * 
 * @param nh Node handle
 */
void param_setup(const ros::NodeHandle& nh)
{

    // String to store parameter directory
    string simulation_param_dir,
    nominal_param_dir;

    // quad model: '+' or 'x'
    QuadModel quad_model;

    // Reference model control gain
    double kp, kv, kq, kw;

    // Parameter for convex function
    mat31_t bound_sigma, bound_theta;
    double epsilon_sigma, epsilon_theta;

    // parameter for projection operator
    mat33_t Gamma_sigma, Gamma_theta;

    double Gamma_theta1, Gamma_theta2, Gamma_theta3;

    // parameter for low pass filter
    double tau_sigma, tau_theta;

    bound_sigma.setZero();
    bound_theta.setZero();
    Gamma_sigma.setZero();
    Gamma_theta.setZero();

    // Get parameter configuration directory
    // for simulation and nominal model, respectively.
    nh.getParam("simulation_param_dir", simulation_param_dir);
    nh.getParam("nominal_param_dir", nominal_param_dir);

    int quad_model_;

    nh.getParam("quad_model",quad_model_);
    if (quad_model_ == 1)
        quad_model = QuadModel::model1;
    else
        quad_model = QuadModel::model2;

    nh.getParam("simulation_final_time", Tf);
    nh.getParam("discrete_time", dt);

    nh.getParam("kp", kp);
    nh.getParam("kv", kv);
    nh.getParam("kq", kq);
    nh.getParam("kw", kw);

    nh.getParam("b_sigma", bound_sigma(0));
    nh.getParam("e_sigma", epsilon_sigma);

    nh.getParam("b_theta", bound_theta(0));
    nh.getParam("e_theta", epsilon_theta);

    nh.getParam("Gamma_sigma", Gamma_sigma(0));
    nh.getParam("Gamma_theta1", Gamma_theta1);
    nh.getParam("Gamma_theta2", Gamma_theta2);
    nh.getParam("Gamma_theta3", Gamma_theta3);
    Gamma_theta(0) = Gamma_theta1;
    Gamma_theta(4) = Gamma_theta2;
    Gamma_theta(8) = Gamma_theta3;


    nh.getParam("tau_sigma", tau_sigma);
    nh.getParam("tau_theta", tau_theta);

    nh.getParam("line_width", line_width);
    nh.getParam("font_size", font_size);

    for(size_t i = 0; i < 3; i++)
    {
        bound_sigma(i) = bound_sigma(0);
        bound_theta(i) = bound_theta(0);

        Gamma_sigma(i*4) = Gamma_sigma(0);
        Gamma_theta(i*4) = Gamma_theta(0);
    }

    // Declare inertial parameters for simulation
    // and nominal model.
    inertial_param_t simulation_inertial_param;
    inertial_param_t nominal_inertial_param;

    // Declare lift and moment coefficients
    aero_coeff_t aero_coeff;

    // Declare arm length
    double l;

    ReadConfig read_simulation_param_obj = 
    ReadConfig(simulation_param_dir);

    ReadConfig read_nominal_param_obj = 
    ReadConfig(nominal_param_dir);

    // Write the parameter by reference.
    read_simulation_param_obj.get_param(simulation_inertial_param, 
    aero_coeff, l);

    read_nominal_param_obj.get_param(nominal_inertial_param);

    // Compute simulation time step
    N = static_cast<int>(Tf/dt);

    // Allocate vector by the simulation type step
    variable_capacity_reserve(N);

    simulation_time.reserve(N);

    for(size_t i = 0; i < N; i++)
    {
        simulation_time.push_back((i+1)*dt);
    }

    assert(dt > std::numeric_limits<double>::min());
    assert(simulation_time.capacity() == N);

    print_parameter_setup(simulation_inertial_param,
        aero_coeff,
        bound_theta, epsilon_theta,
        bound_theta, epsilon_theta,
        Gamma_sigma, Gamma_theta,
        tau_sigma, tau_theta);

    // Simulation model object to test estimation performance
    simulation_model_ptr 
    = new SimulationModel(quad_model, 
    aero_coeff, 
    simulation_inertial_param, 
    l);

    reference_model_ptr = 
    new RefModel(nominal_inertial_param,
    kp, kv, kq, kw);

    disturbance_est_ptr
    = new DisturbanceEstimator(nominal_inertial_param,
        bound_sigma, epsilon_sigma, 
        bound_theta, epsilon_theta, 
        Gamma_sigma, Gamma_theta,
        tau_sigma, tau_theta);

    sigma_est_noisy.setZero();
    theta_est_noisy.setZero();

    sigma_est_lpf.setZero();
    theta_est_lpf.setZero();

    p_state_prev.setZero();
    v_state_prev.setZero();

    q_state_prev.setIdentity();
    w_state_prev.setZero();

    p_ref.setZero();
    v_ref.setZero();
    q_ref.setIdentity();
    w_ref.setZero();

    initialilze_temporary_data_to_store_min_max();

}

/**
 * @brief print parameter setup
 * 
 * @param bound_theta 
 * @param epsilon_theta 
 * @param bound_sigma 
 * @param epsilon_sigma 
 * @param Gamma_sigma 
 * @param Gamma_theta 
 * @param tau_sigma 
 * @param tau_theta 
 */
void print_parameter_setup(const inertial_param_t& inertial_param, 
    const aero_coeff_t& aero_param,
    const mat31_t &bound_theta, const double &epsilon_theta, 
    const mat31_t &bound_sigma, const double &epsilon_sigma, 
    const mat33_t &Gamma_sigma, const mat33_t &Gamma_theta, 
    const double &tau_sigma, const double &tau_theta)
{
    cout << "***************************************" << endl;
    cout << "Inertial parameter setup" << endl;
    cout << "mass: " << inertial_param.m << endl;
    cout << "MOI: " << endl;
    cout << inertial_param.J <<endl;
    cout << endl;
    cout << "COM offset: ";
    for(size_t i = 0; i < inertial_param.r_offset.size(); i++)
        cout << inertial_param.r_offset(i) << " ";
    cout<<endl;


    cout << "***************************************" << endl;
    cout << "Aero parameter setup" << endl;
    cout << "lift coeff: " << aero_param.lift_coeff << endl;
    cout << "moment coeff: " << aero_param.moment_coeff << endl;

    cout << "***************************************" << endl;
    cout << "Convex function setup" << endl;
    cout << "Bound (trans): " << endl;
    for(size_t i = 0; i < bound_sigma.size(); i++)
        cout << bound_sigma(i) << " ";
    cout<<endl;
    cout << "Epsil (trans): " << epsilon_sigma << endl;
    cout << endl;
    cout << "Bound (orien): " << endl;
    for(size_t i = 0; i < bound_theta.size(); i++)
        cout << bound_theta(i) << " ";
    cout << endl;
    cout << "Epsil (orien): " << epsilon_theta << endl;


    cout << "***************************************" << endl;

    cout << "Gamma projection setup" << endl;
    cout << "Gamma Proj (trans): " << endl;
    cout << Gamma_sigma << endl;
    cout << endl;
    cout << "Gamma Proj (orien): " << endl;
    cout << Gamma_theta << endl;

    cout << "***************************************" << endl;

    cout << "Low pass filter setup" << endl;
    cout << "tau (trans): "<< tau_sigma << endl;
    cout << "tau (orien): "<< tau_theta << endl;

    cout << "***************************************" << endl;

    cout << "Simulation time setup" << endl;
    cout<<"Final Time: "<< Tf <<endl;
    cout<<"Discrete time: "<< dt <<endl;
    cout<<"Simulation step: "<< N <<endl;

    cout << "***************************************" << endl;
}


/**
 * @brief play simulation model
 * 
 * @param rpm_ rotor rpm
 * @param sigma_ext_ disturbance (trans)
 * @param theta_ext_ disturbance (orien)
 * @param simulation_time_ simulation time
 */
void play_simulation_model(const mat41_t &rpm_, 
const mat31_t &sigma_ext_, const mat31_t &theta_ext_, 
const double &simulation_time_)
{

    assert(rpm_.size() == 4);
    assert(sigma_ext_.size() == 3);
    assert(theta_ext_.size() == 3);

    simulation_model_ptr->set_control_input(rpm_);
    simulation_model_ptr->set_disturbance(sigma_ext_, theta_ext_);
    simulation_model_ptr->set_time(simulation_time_);
    simulation_model_ptr->integrate();
    
}


/**
 * @brief Allocate variables by the size of N
 * 
 * @param N_ 
 */
inline void variable_capacity_reserve(const int &N_)
{
    // simulation time
    simulation_time.reserve(N_);

    // Simulation state
    // ************************************************
    // State position variables
    x_state.reserve(N_);
    y_state.reserve(N_);
    z_state.reserve(N_);

    x_state.push_back(0);
    y_state.push_back(0);
    z_state.push_back(0);

    // State velocity variables
    vx_state.reserve(N_);
    vy_state.reserve(N_);
    vz_state.reserve(N_);

    vx_state.push_back(0);
    vy_state.push_back(0);
    vz_state.push_back(0);
    
    // State quaternion variables
    qw_state.reserve(N_);
    qx_state.reserve(N_);
    qy_state.reserve(N_);
    qz_state.reserve(N_);

    qw_state.push_back(1);
    qx_state.push_back(0);
    qy_state.push_back(0);
    qz_state.push_back(0);

    // State angular velocity variables
    wx_state.reserve(N_);
    wy_state.reserve(N_);
    wz_state.reserve(N_);

    wx_state.push_back(0);
    wy_state.push_back(0);
    wz_state.push_back(0);

    // ************************************************
    // Reference model state
    // Reference position variables
    x_ref.reserve(N_);
    y_ref.reserve(N_);
    z_ref.reserve(N_);

    x_ref.push_back(0);
    y_ref.push_back(0);
    z_ref.push_back(0);

    // Reference velocity variables
    vx_ref.reserve(N_);
    vy_ref.reserve(N_);
    vz_ref.reserve(N_);

    vx_ref.push_back(0);
    vy_ref.push_back(0);
    vz_ref.push_back(0);
    
    // REference quaternion variables
    qw_ref.reserve(N_);
    qx_ref.reserve(N_);
    qy_ref.reserve(N_);
    qz_ref.reserve(N_);

    qw_ref.push_back(1);
    qx_ref.push_back(0);
    qy_ref.push_back(0);
    qz_ref.push_back(0);

    // Reference angular velocity variables
    wx_ref.reserve(N_);
    wy_ref.reserve(N_);
    wz_ref.reserve(N_);

    wx_ref.push_back(0);
    wy_ref.push_back(0);
    wz_ref.push_back(0);
     
    // ************************************************
    // Disturbance terms

    sigma_ext_x.reserve(N_);
    sigma_ext_y.reserve(N_);
    sigma_ext_z.reserve(N_);

    sigma_ext_x.push_back(0);
    sigma_ext_y.push_back(0);
    sigma_ext_z.push_back(0);
    
    sigma_est_x.reserve(N_);
    sigma_est_y.reserve(N_);
    sigma_est_z.reserve(N_);

    sigma_est_x.push_back(0);
    sigma_est_y.push_back(0);
    sigma_est_z.push_back(0);
    
    sigma_est_lpf_x.reserve(N_);
    sigma_est_lpf_y.reserve(N_);
    sigma_est_lpf_z.reserve(N_);

    sigma_est_lpf_x.push_back(0);
    sigma_est_lpf_y.push_back(0);
    sigma_est_lpf_z.push_back(0);
    
    theta_ext_x.reserve(N_);
    theta_ext_y.reserve(N_);
    theta_ext_z.reserve(N_);

    theta_ext_x.push_back(0);
    theta_ext_y.push_back(0);
    theta_ext_z.push_back(0);

    theta_est_x.reserve(N_);
    theta_est_y.reserve(N_);
    theta_est_z.reserve(N_);

    theta_est_x.push_back(0);
    theta_est_y.push_back(0);
    theta_est_z.push_back(0);

    theta_est_lpf_x.reserve(N_);
    theta_est_lpf_y.reserve(N_);
    theta_est_lpf_z.reserve(N_);

    theta_est_lpf_x.push_back(0);
    theta_est_lpf_y.push_back(0);
    theta_est_lpf_z.push_back(0);

}

/**
 * @brief Put simulation state in the order of position, 
 * linear velocity, quaternion, and angular velocity
 * 
 * @param position 
 * @param linear_velocity 
 * @param quaternion 
 * @param angular_velocity 
 */
inline void demux_simulation_state(const mat31_t &position, 
const mat31_t &linear_velocity, 
const quat_t &quaternion, 
const mat31_t &angular_velocity)
{
    demux_vec3(position, x_state, y_state, z_state);
    demux_vec3(linear_velocity, vx_state, vy_state, vz_state);
    demux_quat(quaternion, qw_state, qx_state, qy_state, qz_state);
    demux_vec3(angular_velocity, wx_state, wy_state, wz_state);
}

/**
 * @brief Put reference state in the order of position, 
 * linear velocity, quaternion, and angular velocity
 * 
 * @param position 
 * @param linear_velocity 
 * @param quaternion 
 * @param angular_velocity 
 */
inline void demux_reference_state(const mat31_t &position, 
const mat31_t &linear_velocity, 
const quat_t &quaternion, 
const mat31_t &angular_velocity)
{
    demux_vec3(position, x_ref, y_ref, z_ref);
    demux_vec3(linear_velocity, vx_ref, vy_ref, vz_ref);
    demux_quat(quaternion, qw_ref, qx_ref, qy_ref, qz_ref);
    demux_vec3(angular_velocity, wx_ref, wy_ref, wz_ref);
}

inline void demux_disturbance_ext(const mat31_t &sigma_, 
const mat31_t &theta_)
{
    demux_vec3(sigma_, sigma_ext_x, sigma_ext_y, sigma_ext_z);
    demux_vec3(theta_, theta_ext_x, theta_ext_y, theta_ext_z);
}

inline void demux_disturbance_est_noisy(const mat31_t &sigma_, 
const mat31_t &theta_)
{
    demux_vec3(sigma_, sigma_est_x, sigma_est_y, sigma_est_z);
    demux_vec3(theta_, theta_est_x, theta_est_y, theta_est_z);
}

inline void demux_disturbance_est_filtered(const mat31_t &sigma_, 
const mat31_t &theta_)
{
    demux_vec3(sigma_, sigma_est_lpf_x, sigma_est_lpf_y, sigma_est_lpf_z);
    demux_vec3(theta_, theta_est_lpf_x, theta_est_lpf_y, theta_est_lpf_z);
}

/**
 * @brief Demux 3 dimensional vector into comp_x, comp_y, comp_z
 * 
 * @param v 3 dim vector to demux
 * @param comp_x 
 * @param comp_y 
 * @param comp_z 
 */
inline void demux_vec3(const mat31_t &v, 
vector<double> &comp_x, vector<double> &comp_y, vector<double> &comp_z)
{
    comp_x.push_back(v(0));
    comp_y.push_back(v(1));
    comp_z.push_back(v(2));
}

/**
 * @brief Demux quaternion into w, x, y, and z
 * 
 * @param q quaternion to demux
 * @param qw 
 * @param qx 
 * @param qy 
 * @param qz 
 */
inline void demux_quat(const quat_t & q, 
vector<double>& qw, vector<double>& qx, vector<double>& qy, vector<double>& qz)
{
    qw.push_back(q.w());
    qx.push_back(q.x());
    qy.push_back(q.y());
    qz.push_back(q.z());
}

/**
 * @brief Print state variables
 * 
 * @param time 
 * @param p_state 
 * @param v_state 
 * @param q_state 
 * @param w_state 
 * @param p_ref 
 * @param v_ref 
 * @param q_ref 
 * @param w_ref 
 * @param theta_ext 
 * @param sigma_ext 
 * @param theta_est_lpf 
 * @param sigma_est_lpf 
 * @param theta_est_noisy 
 * @param sigma_est_noisy 
 */
inline void print_state(const double& time_, 
const mat31_t& p_state_, const mat31_t& v_state_,
const quat_t &q_state_, const mat31_t &w_state_,
const mat31_t &p_ref_, const mat31_t &v_ref_, 
const quat_t &q_ref_, const mat31_t &w_ref_,
const mat31_t &theta_ext_, const mat31_t &sigma_ext_,
const mat31_t &theta_est_lpf_, const mat31_t &sigma_est_lpf_,
const mat31_t &theta_est_noisy_, const mat31_t &sigma_est_noisy_)
{
    cout << "Simulation time: " << time_ << endl;

    cout << "State (p): " << p_state_(0) 
    << " " << p_state_(1) << " " << p_state_(2) << endl;
    
    cout << "State (v): " << v_state_(0) 
    << " " << v_state_(1) << " " << v_state_(2) << endl;

    cout << "State (q): " << q_state_.w() 
    << " " << q_state_.x() << " " << q_state_.y() <<
    " " << q_state_.z() << endl;

    cout << "State (w): " << w_state_(0) 
    << " " << w_state_(1) << " " << w_state_(2) << endl;



    cout << "Reference (p): " << p_ref_(0) 
    << " " << p_ref_(1) << " " << p_ref_(2) << endl;
    
    cout << "Reference (v): " << v_ref_(0) 
    << " " << v_ref_(1) << " " << v_ref_(2) << endl;

    cout << "Reference (q): " << q_ref_.w() 
    << " " << q_ref_.x() << " " << q_ref_.y() <<
    " " << q_ref_.z() << endl;

    cout << "Reference (w): " << w_ref_(0) 
    << " " << w_ref_(1) << " " << w_ref_(2) << endl;

    cout<<endl;

    cout << "Translational disturbance info" << endl;
    cout << "Disturbance ext: "<< sigma_ext_(0) 
    <<" " << sigma_ext_(1) << " " << sigma_ext_(2) << endl;

    cout << "Disturbance est noisy: "<< sigma_est_noisy_(0) 
    <<" " << sigma_est_noisy_(1) << " " << sigma_est_noisy_(2) << endl;

    cout << "Disturbance est filtered: "<< sigma_est_lpf_(0) 
    <<" " << sigma_est_lpf_(1) << " " << sigma_est_lpf_(2) << endl;

    cout<<endl;

    cout << "Orientational disturbance info" << endl;
    cout << "Disturbance ext: "<< theta_ext_(0) 
    <<" " << theta_ext_(1) << " " << theta_ext_(2) << endl;

    cout << "Disturbance est noisy: "<< theta_est_noisy_(0) 
    <<" " << theta_est_noisy_(1) << " " << theta_est_noisy_(2) << endl;

    cout << "Disturbance est filtered: "<< theta_est_lpf_(0) 
    <<" " << theta_est_lpf_(1) << " " << theta_est_lpf_(2) << endl;

    cout << endl;
}

inline void initialilze_temporary_data_to_store_min_max()
{
    x_max = x_min = y_max = y_min = z_max = z_min = 0;

    vx_max = vx_min = vy_max = vy_min = vz_max = vz_min = 0;

    wx_max = wx_min = wy_max = wy_min = wz_max = wz_min = 0;
}

inline void find_max(const double &data, double &dest)
{
    if(data >= dest)
    {
        dest = data;
    }
    else
    {
        return;
    }
}

inline void find_min(const double &data, double &dest)
{
    if(data <= dest)
    {
        dest = data;
    }
    else
    {
        return;
    }
}

#endif