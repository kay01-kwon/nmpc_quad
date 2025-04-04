#include "read_config.hpp"

ReadConfig::ReadConfig(const string &file_name)
:file_name_(file_name)
{
    assert(file_name.size() > 0);
    cout<<"The directory of the yaml file: "
    << file_name_<<endl;
    load_yaml_file();
}

void ReadConfig::get_param(inertial_param_t &inertial_param, 
aero_coeff_t &aero_coeff, 
double &arm_length) const
{
    inertial_param = inertial_param_;
    aero_coeff = aero_coeff_;
    arm_length = arm_length;
}

void ReadConfig::get_param(inertial_param_t &inertial_param) const
{
    inertial_param = inertial_param_;
}

void ReadConfig::load_yaml_file()
{
    try{
        Node config = YAML::LoadFile(file_name_);
        cout<<"Yaml file is loaded."<<endl;

        get_Inertial_param_from_yaml(config);
        get_Aero_coeff_from_yaml(config);
        get_arm_length_from_yaml(config);
    }
    catch(const YAML::BadFile& e)
    {
        std::cerr<<e.msg<<endl;
    }
    catch(const YAML::ParserException& e)
    {
        std::cerr<<e.msg<<endl;
    }
}

void ReadConfig::get_Inertial_param_from_yaml(const Node &config)
{
    auto inertial_param_yaml = config["inertial_param"];
    
    double Jxx, Jxy, Jxz, Jyy, Jyz, Jzz;
    double rx, ry, rz;

    assert(inertial_param_yaml > 0);

    assert(inertial_param_yaml["mass"]["m"] != 0);

    assert(inertial_param_yaml["MOI"]["Jxx"] != 0);
    assert(inertial_param_yaml["MOI"]["Jxy"] != 0);
    assert(inertial_param_yaml["MOI"]["Jxz"] != 0);
    assert(inertial_param_yaml["MOI"]["Jyy"] != 0);
    assert(inertial_param_yaml["MOI"]["Jyz"] != 0);
    assert(inertial_param_yaml["MOI"]["Jzz"] != 0);

    assert(inertial_param_yaml["offset"]["x"] != 0);
    assert(inertial_param_yaml["offset"]["y"] != 0);
    assert(inertial_param_yaml["offset"]["z"] != 0);

    // Get mass information
    inertial_param_.m = inertial_param_yaml["mass"]["m"].as<double>();



    // Get MOI information
    Jxx = inertial_param_yaml["MOI"]["Jxx"].as<double>();
    Jxy = inertial_param_yaml["MOI"]["Jxy"].as<double>();
    Jxz = inertial_param_yaml["MOI"]["Jxz"].as<double>();
    Jyy = inertial_param_yaml["MOI"]["Jyy"].as<double>();
    Jyz = inertial_param_yaml["MOI"]["Jyz"].as<double>();
    Jzz = inertial_param_yaml["MOI"]["Jzz"].as<double>();
    
    inertial_param_.J <<    Jxx, Jxy, Jxz,
                            Jxy, Jyy, Jyz,
                            Jxz, Jyz, Jzz;

    // Get offset information
    rx = inertial_param_yaml["offset"]["x"].as<double>();
    ry = inertial_param_yaml["offset"]["y"].as<double>();
    rz = inertial_param_yaml["offset"]["z"].as<double>();
    
    inertial_param_.r_offset << rx, ry, rz;


    // cout << "***************************************" << endl;
    // cout << "Mass: " << inertial_param_.m << endl;
    // cout << "MOI matrix: "<<endl;
    // cout<<inertial_param_.J<<endl;
    // cout << "offset: "<<endl;
    // cout<<inertial_param_.r_offset<<endl;

}

void ReadConfig::get_Aero_coeff_from_yaml(const Node &config)
{
    auto aero_coeff_yaml = config["aerodynamics_param"];

    double C_l, C_m;

    assert(aero_coeff_yaml != 0);
    assert(aero_coeff_yaml["C_l"] != 0);
    assert(aero_coeff_yaml["C_m"] != 0);

    C_l = aero_coeff_yaml["C_l"].as<double>();
    C_m = aero_coeff_yaml["C_m"].as<double>();

    aero_coeff_.lift_coeff = C_l;
    aero_coeff_.moment_coeff = C_m;

    // cout << "lift coefficient: " << C_l << endl;
    // cout << "moment coefficient: " << C_m << endl;
}

void ReadConfig::get_arm_length_from_yaml(const Node &config)
{
    auto arm_length_yaml = config["length_param"];

    assert(arm_length_yaml != 0);
    assert(arm_length_yaml["l"] != 0);


    arm_length_ = arm_length_yaml["l"].as<double>();
    // cout << "Arm length: " << arm_length_ << endl;
    // cout << "***************************************" << endl;

}