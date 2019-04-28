#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components)
{
	this->n_components=n_components;
}

void PCA::fit(Matrix X)
{
	this->X=X;
	//tire esto pero ni idea, que se supone que hace esta funcion??
}


MatrixXd PCA::transform(SparseMatrix X)
{	
	MatrixXd res;
	MatrixXd copia =X;
	
	Eigen::VectorXd medias=copia.colwise().mean();
	
	copia.rowwise()-=medias.transpose(); //centro los valores de las medias
	
	MatrixXd M=copia.transpose()*copia; 
	
	M=M/(X.rows()-1);//creo la matriz de covarianza
	
	std::pair<Eigen::VectorXd, Matrix> base = get_first_eigenvalues(M,M.rows()); //la diagonalizo
	
	MatrixXd V=base.second; //me quedo con la matriz con autovectores en columna (los primeros son los mas significativos, los de mayor autovalor asociado). Notar es tambien la matriz de cambio de base
	
	res=X*(V.leftCols(this->n_components)); //hago el cambio de base teniendo en cuenta las primeras componentes principales
	
	return res;
  //throw std::runtime_error("Sin implementar");
}
