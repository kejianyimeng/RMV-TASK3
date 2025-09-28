#include <iostream>
#include <vector>
#include <ceres/ceres.h>

struct TrajectoryCostFunction {
    TrajectoryCostFunction(double t, double x_obs, double y_obs, double x0, double y0)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0) {}
    
    template<typename T>
    bool operator()(const T* const vx0, const T* const vy0, const T* const g, const T* const k, T* residual) const {
        T dt = T(t_);
        
        // x(t) = x0 + (vx0/k) * (1 - exp(-k*dt))
        T x_pred = T(x0_) + (vx0[0] / k[0]) * (T(1.0) - exp(-k[0] * dt));
        
        // y(t) = y0 + ((vy0 + g/k)/k) * (1 - exp(-k*dt)) - (g/k)*dt
        T term1 = (vy0[0] + g[0] / k[0]) / k[0];
        T term2 = T(1.0) - exp(-k[0] * dt);
        T term3 = (g[0] / k[0]) * dt;
        T y_pred = T(y0_) + term1 * term2 - term3;
        
        residual[0] = x_pred - T(x_obs_);
        residual[1] = y_pred - T(y_obs_);
        
        return true;
    }
    
private:
    double t_, x_obs_, y_obs_, x0_, y0_;
};

class ParameterBoundary : public ceres::Manifold {
public:
    ParameterBoundary(double lower_bound, double upper_bound) 
        : lower_bound_(lower_bound), upper_bound_(upper_bound) {}
    
    virtual int AmbientSize() const { return 1; }
    virtual int TangentSize() const { return 1; }
    
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        x_plus_delta[0] = x[0] + delta[0];
        return true;
    }
    
    virtual bool PlusJacobian(const double* x, double* jacobian) const {
        jacobian[0] = 1.0;
        return true;
    }
    
    virtual bool Minus(const double* y, const double* x, double* y_minus_x) const {
        y_minus_x[0] = y[0] - x[0];
        return true;
    }
    
    virtual bool MinusJacobian(const double* x, double* jacobian) const {
        jacobian[0] = 1.0;
        return true;
    }
    
    virtual bool ValidateParameters(const double* parameters) const {
        return (parameters[0] >= lower_bound_ && parameters[0] <= upper_bound_);
    }
    
private:
    double lower_bound_, upper_bound_;
};

int main() {
    // 数据点
    std::vector<std::pair<double, double>> data_points = {
        {164.032, 126.853}, {168.731, 122.089}, {170.847, 116.95}, {174.851, 112.233},
        {180.969, 106.908}, {184.243, 100.82}, {188.064, 96.1396}, {194.149, 90.7667},
        {198.196, 84.1667}, {200.851, 80.7667}, {206, 78.1875}, {208.757, 74.1805},
        {214, 68.9485}, {218, 64.9485}, {222.847, 60.9497}, {226.861, 59.9866},
        {230.889, 52.9253}, {236.301, 48.6879}, {238.149, 46.7667}, {242.757, 42.8195},
        {246.148, 42.1832}, {250.064, 38.8604}, {254.955, 34.9988}, {260.769, 32.8734},
        {264.032, 30.8533}, {268.847, 26.9497}, {272.803, 26.7837}, {276.803, 24.2163},
        {280.738, 22.2032}, {283.968, 20.1209}, {286.819, 18.907}, {291.961, 18.0265},
        {296, 16.1875}, {300.857, 15}, {304.131, 11.9685}, {308.852, 12.1832},
        {313, 10.7612}, {316.916, 12.0464}, {320.148, 10.8168}, {322.852, 10.1832},
        {327.874, 10.1171}, {331.874, 10.1171}, {334.851, 10.2333}, {340.851, 8.76667},
        {344.88, 10.1432}, {348.88, 12.1432}, {352.852, 12.1832}, {358.112, 14.0756},
        {360.107, 12.2584}, {364.818, 14.1432}, {368.852, 14.8168}, {374, 18.2388},
        {376, 18.2388}, {380.764, 21}, {386.207, 20.125}, {388.869, 23.9394},
        {392.139, 26.0159}, {396.819, 26.093}, {400.852, 28.8168}, {404.096, 32.0498},
        {408.081, 34.9674}, {412.149, 38.2333}, {416.182, 38.1432}, {420.182, 42.1432},
        {424.851, 46.2333}, {428.818, 48.8568}, {432.847, 50.9497}, {434.197, 56.7837},
        {439, 60.1875}, {444.149, 62.7667}, {448.143, 68}, {453, 70.7612},
        {454.966, 76.0474}, {460.24, 79.9556}, {462.12, 84.1432}, {468.159, 88.0898},
        {470.861, 93.0134}, {475, 98.2388}, {478.947, 104.151}, {482.851, 108.767},
        {486.197, 116.216}, {490.032, 118.853}, {494.032, 124.853}, {498.149, 130.233},
        {502.185, 134.999}, {506.197, 142.784}, {508.236, 148}, {514.851, 154.767},
        {516.196, 160.833}, {520.869, 167.939}, {524.053, 174.089}, {524.904, 180.05},
        {532.749, 186.108}, {536.852, 192.817}, {541, 202.188}, {543, 208.188},
        {548.279, 214.934}, {550.867, 222.02}, {556.112, 230.924}, {560.064, 236.909},
        {562.851, 246.233}, {567.032, 254.121}, {570.243, 262.82}, {574.196, 270.167},
        {578.177, 279.046}, {582.148, 286.183}, {584.888, 296.076}, {588.149, 306.233},
        {592.936, 314.14}, {596.852, 322.817}, {598.852, 332.817}, {602.851, 342.233},
        {608.181, 350.907}, {610.851, 360.767}, {616.197, 368.784}, {618.851, 380.767},
        {622.851, 390.233}, {626.112, 400.124}, {630.197, 408.784}, {634.197, 420.216},
        {638.064, 430.14}, {642.197, 440.784}, {642.804, 452.167}, {648.149, 462.767},
        {652.149, 472.767}, {654.851, 482.767}, {659.991, 495.942}, {663.914, 506.917},
        {668.053, 518.879}, {672.064, 530.091}, {674.053, 540.879}, {678.112, 552.876},
        {680.725, 566.158}, {684.74, 576.844}, {688.97, 590.071}, {692.136, 600.738},
        {694.939, 614.805}, {700.179, 628.867}, {702.731, 640.089}, {706.159, 652.09},
        {710.112, 664.876}, {714.064, 678.86}, {718.045, 690.916}, {722.888, 704.876}
    };
    
    // 初始位置 (第一帧)
    double x0 = data_points[0].first;
    double y0 = data_points[0].second;
    
    // 初始参数估计
    double vx0 = 300.0;  // 初始x方向速度估计
    double vy0 = -50.0;  // 初始y方向速度估计
    double g = 500.0;    // 重力加速度初始估计
    double k = 0.1;      // 阻力系数初始估计
    
    // 创建优化问题
    ceres::Problem problem;
    
    // 添加参数边界约束
    problem.AddParameterBlock(&g, 1, new ParameterBoundary(100.0, 1000.0));
    problem.AddParameterBlock(&k, 1, new ParameterBoundary(0.01, 1.0));
    
    // 添加所有数据点的残差项
    for (size_t i = 1; i < data_points.size(); ++i) {
        double t = i / 60.0;  // 时间 (秒)，FPS=60
        double x_obs = data_points[i].first;
        double y_obs = data_points[i].second;
        
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<TrajectoryCostFunction, 2, 1, 1, 1, 1>(
                new TrajectoryCostFunction(t, x_obs, y_obs, x0, y0));
        
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &vx0, &vy0, &g, &k);
    }
    
    // 配置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-8;
    options.parameter_tolerance = 1e-8;
    
    // 求解
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 输出结果
    std::cout << summary.FullReport() << std::endl;
    std::cout << "拟合结果:" << std::endl;
    std::cout << "初始速度 vx0 = " << vx0 << " px/s" << std::endl;
    std::cout << "初始速度 vy0 = " << vy0 << " px/s" << std::endl;
    std::cout << "重力加速度 g = " << g << " px/s²" << std::endl;
    std::cout << "阻力系数 k = " << k << " 1/s" << std::endl;
    std::cout << "初始位置 x0 = " << x0 << " px" << std::endl;
    std::cout << "初始位置 y0 = " << y0 << " px" << std::endl;
    
    // 计算拟合误差
    double total_error = 0.0;
    for (size_t i = 0; i < data_points.size(); ++i) {
        double t = i / 60.0;
        double dt = t;
        
        double x_pred = x0 + (vx0 / k) * (1.0 - exp(-k * dt));
        double term1 = (vy0 + g / k) / k;
        double term2 = 1.0 - exp(-k * dt);
        double term3 = (g / k) * dt;
        double y_pred = y0 + term1 * term2 - term3;
        
        double error_x = x_pred - data_points[i].first;
        double error_y = y_pred - data_points[i].second;
        total_error += sqrt(error_x * error_x + error_y * error_y);
    }
    
    double avg_error = total_error / data_points.size();
    std::cout << "平均拟合误差: " << avg_error << " px" << std::endl;
    
    return 0;
}