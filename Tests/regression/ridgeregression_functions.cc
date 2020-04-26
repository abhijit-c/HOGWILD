#include <Eigen/Core>

double
obj(Eigen::VectorXd x,
    long long int i  ,
    Eigen::MatrixXd A,
    Eigen::VectorXd b,
    double lambda)
{
  if (i == -1)
  {
    return (A*x-b).squaredNorm() + lambda*x.squaredNorm();
  }
  else
  {
    return (A.row(i)*x - b(i))^2 + lambda*x(i)^2;
  }
}
double
obj_grad(Eigen::VectorXd x,
         long long int i  ,
         Eigen::MatrixXd A,
         Eigen::VectorXd b,
         double lambda)
{
  if (i == -1)
  {
    return 2*(A*x-b) + 2*lambda*2*x;
  }
  else
  {
    return 2*(A.row(i)*x - b(i)) + 2*lambda*2*x(i);
  }
}
