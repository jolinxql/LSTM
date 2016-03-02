#ifndef NET_H___
#define NET_H___
#include <string>
#include <Eigen/Dense>
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;

class Net
{
public:
	int inputlen;
	int outputlen;
	string gateActiFunc;
	string statActiFunc;
	double weightRange;
	double learningRate;	//alpha
	double l2_reg;			//weightPenalty;
	double momentum;
	MatrixXd *Xs;
	MatrixXd *Ys;

#pragma region Mem
	MatrixXd prev_mc;
	MatrixXd prev_mh;
	MatrixXd mi;
	MatrixXd mai;
	MatrixXd mf;
	MatrixXd maf;
	MatrixXd mc;
	MatrixXd mac;
	MatrixXd mgac;
	MatrixXd mo;
	MatrixXd mao;
	MatrixXd mgc;
	MatrixXd mh;

	MatrixXd d_mi;
	MatrixXd d_mai;
	MatrixXd d_mf;
	MatrixXd d_maf;
	MatrixXd d_mc;
	MatrixXd d_mac;
	MatrixXd d_mo;
	MatrixXd d_mao;
	MatrixXd d_mh;
	MatrixXd d_x;
#pragma endregion

#pragma region W
			//i(t)
	MatrixXd wix;
	MatrixXd wih;
	MatrixXd wic;

	//f(t)
	MatrixXd wfx;
	MatrixXd wfh;
	MatrixXd wfc;

	//c(t)
	MatrixXd wcx;
	MatrixXd wch;

	//o(t)
	MatrixXd wox;
	MatrixXd woh;
	MatrixXd woc;
#pragma endregion

#pragma region d_W
	MatrixXd d_wix;
	MatrixXd d_wih;
	MatrixXd d_wic;

	MatrixXd d_wfx;
	MatrixXd d_wfh;
	MatrixXd d_wfc;

	MatrixXd d_wcx;
	MatrixXd d_wch;

	MatrixXd d_wox;
	MatrixXd d_woh;
	MatrixXd d_woc;
#pragma endregion

#pragma region this_d_W
	MatrixXd this_d_wix;
	MatrixXd this_d_wih;
	MatrixXd this_d_wic;

	MatrixXd this_d_wfx;
	MatrixXd this_d_wfh;
	MatrixXd this_d_wfc;

	MatrixXd this_d_wcx;
	MatrixXd this_d_wch;

	MatrixXd this_d_wox;
	MatrixXd this_d_woh;
	MatrixXd this_d_woc;
#pragma endregion

#pragma region vd_W
	MatrixXd vd_wix;
	MatrixXd vd_wih;
	MatrixXd vd_wic;

	MatrixXd vd_wfx;
	MatrixXd vd_wfh;
	MatrixXd vd_wfc;

	MatrixXd vd_wcx;
	MatrixXd vd_wch;

	MatrixXd vd_wox;
	MatrixXd vd_woh;
	MatrixXd vd_woc;
#pragma endregion

	void mycoutmat(MatrixXd *out);
	void Setup(int _inputlen
		, int _outputlen
		, string _gateActiFunc
		, string _statActiFunc
		, double  _weightRange
		, double _learningRate
		, double _l2_reg
		, double _momentum);
	void ActiFunc(MatrixXd * x, int xt, MatrixXd * y, int yt, string gateActiFunc);
	void ActiFunc_d(MatrixXd * x, int xt, MatrixXd * y, int yt, string gateActiFunc);
	void FF();
	void BPTT(MatrixXd & e);
	void Update(int thisBatchSize);
	void UpdateEmbedding(MatrixXd *XIds, vector<vector<double> > *embedding);
};
#endif