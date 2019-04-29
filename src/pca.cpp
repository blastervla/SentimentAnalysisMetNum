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
	MatrixXd copia =X;
	
	Eigen::VectorXd medias=copia.colwise().mean();
	
	copia.rowwise()-=medias.transpose(); //centro los valores de las medias
	
	MatrixXd M=copia.transpose()*copia; 
	
	M=M/(X.rows()-1);//creo la matriz de covarianza
	
	std::pair<Eigen::VectorXd, Matrix> autovects = get_first_eigenvalues(M,this->n_components); //me quedo con los primeros autovects (las n_components componentes principales)
	
	MatrixXd V=autovects.second; //me quedo con la matriz con autovectores en columna (los primeros son los mas significativos, los de mayor autovalor asociado). Notar es tambien la matriz de cambio de base
	
	this->base = V;	
}


MatrixXd PCA::transform(SparseMatrix X)
{	
	MatrixXd res = X*(this->base);//hago el cambio de base teniendo en cuenta las primeras componentes principales
	return res;
}
