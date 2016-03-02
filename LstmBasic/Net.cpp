#include "Net.h"
void Net::mycoutmat(MatrixXd * out)
{
	cout << (*out) << endl;
	getchar();
}
void Net::Setup(int _inputlen
	, int _outputlen
	, string _gateActiFunc
	, string _statActiFunc
	, double  _weightRange
	, double _learningRate
	, double _l2_reg
	, double _momentum)
{
	inputlen = _inputlen;
	outputlen = _outputlen;
	gateActiFunc = _gateActiFunc;
	statActiFunc = _statActiFunc;
	weightRange = _weightRange;
	learningRate = _learningRate;
	l2_reg = _l2_reg;
	momentum = _momentum;

	wix = (MatrixXd::Random(outputlen, inputlen + 1) - MatrixXd::Ones(outputlen, inputlen + 1)*0.5) / inputlen * weightRange;
	wih = (MatrixXd::Random(outputlen, outputlen) - MatrixXd::Ones(outputlen, outputlen)*0.5) / outputlen * weightRange;
	wic = (MatrixXd::Random(outputlen, 1) - MatrixXd::Ones(outputlen, 1)*0.5) / outputlen / 5;

	wfx = (MatrixXd::Random(outputlen, inputlen + 1) - MatrixXd::Ones(outputlen, inputlen + 1)*0.5) / inputlen * weightRange;
	wfh = (MatrixXd::Random(outputlen, outputlen) - MatrixXd::Ones(outputlen, outputlen)*0.5) / outputlen * weightRange;
	wfc = (MatrixXd::Random(outputlen, 1) - MatrixXd::Ones(outputlen, 1)*0.5) / outputlen / 5;

	wcx = (MatrixXd::Random(outputlen, inputlen + 1) - MatrixXd::Ones(outputlen, inputlen + 1)*0.5) / inputlen * weightRange;
	wch = (MatrixXd::Random(outputlen, outputlen) - MatrixXd::Ones(outputlen, outputlen)*0.5) / outputlen * weightRange;

	wox = (MatrixXd::Random(outputlen, inputlen + 1) - MatrixXd::Ones(outputlen, inputlen + 1)*0.5) / inputlen * weightRange;
	woh = (MatrixXd::Random(outputlen, outputlen) - MatrixXd::Ones(outputlen, outputlen)*0.5) / outputlen * weightRange;
	woc = (MatrixXd::Random(outputlen, 1) - MatrixXd::Ones(outputlen, 1)*0.5) / outputlen / 5;

	if (momentum > 0)
	{
		vd_wix = MatrixXd::Zero(outputlen, inputlen + 1);
		vd_wih = MatrixXd::Zero(outputlen, outputlen);
		vd_wic = MatrixXd::Zero(outputlen, 1);

		vd_wfx = MatrixXd::Zero(outputlen, inputlen + 1);
		vd_wfh = MatrixXd::Zero(outputlen, outputlen);
		vd_wfc = MatrixXd::Zero(outputlen, 1);

		vd_wcx = MatrixXd::Zero(outputlen, inputlen + 1);
		vd_wch = MatrixXd::Zero(outputlen, outputlen);

		vd_wox = MatrixXd::Zero(outputlen, inputlen + 1);
		vd_woh = MatrixXd::Zero(outputlen, outputlen);
		vd_woc = MatrixXd::Zero(outputlen, 1);
	}

	d_wix = MatrixXd::Zero(outputlen, inputlen + 1);
	d_wih = MatrixXd::Zero(outputlen, outputlen);
	d_wic = MatrixXd::Zero(outputlen, 1);

	d_wfx = MatrixXd::Zero(outputlen, inputlen + 1);
	d_wfh = MatrixXd::Zero(outputlen, outputlen);
	d_wfc = MatrixXd::Zero(outputlen, 1);

	d_wcx = MatrixXd::Zero(outputlen, inputlen + 1);
	d_wch = MatrixXd::Zero(outputlen, outputlen);

	d_wox = MatrixXd::Zero(outputlen, inputlen + 1);
	d_woh = MatrixXd::Zero(outputlen, outputlen);
	d_woc = MatrixXd::Zero(outputlen, 1);
}

void Net::ActiFunc(MatrixXd *x, int xt, MatrixXd *y, int yt, string ActiFunc)
{
	if (strcmp("sigm", ActiFunc.c_str()) == 0)
	{
		(*y).row(yt) = 1 / (1 + (-(*x).row(xt)).array().exp());
	}
	else if (strcmp("tanh_opt", gateActiFunc.c_str()) == 0)
	{
		ArrayXXd pos = ((*x).array() * 2 / 3).exp();
		ArrayXXd neg = (-(*x).array() * 2 / 3).exp();
		ArrayXXd tanh = (pos - neg) / (pos + neg);
		(*y) = 1.7159*tanh;
	}
	else if (strcmp("linear", ActiFunc.c_str()) == 0)
	{
		(*y) = (*x);
	}
	//else if (strcmp("softmax", ActiFunc.c_str()))
	//{
	//	(*y) = 1 / (1 + -x.array().exp());
	//}
}

void Net::ActiFunc_d(MatrixXd *x, int xt, MatrixXd *y, int yt, string ActiFunc)
{
	if (strcmp("sigm", ActiFunc.c_str())==0)
	{
		(*y).row(yt) = (*x).row(xt).array()*(1 - (*x).row(xt).array());
	}
	else if (strcmp("tanh_opt", ActiFunc.c_str()) == 0)
	{
		(*y) = 1.7159 * 2 / 3 * (1 - 1 / (1.7159*1.7159)*(*x).array()*(*x).array());
	}
	else if (strcmp("linear", ActiFunc.c_str()) == 0)
	{
		(*y) = MatrixXd::Ones((*x).rows(),(*x).cols());
	}
	//else if (strcmp("softmax", ActiFunc.c_str()))
	//{
	//	(*y) = 1 / (1 + -x.array().exp());
	//}
}

void Net::FF()
{
	int m = (*Xs).rows();
	int n = (*Xs).cols();
	if (n != inputlen + 1)
	{
		printf("Error! Input length is not correspond with the lstm config.");
		getchar();
		exit(0);
	}
	prev_mc = MatrixXd::Zero(1, outputlen);
	prev_mh = MatrixXd::Zero(1, outputlen);

	mi = MatrixXd::Zero(m, outputlen);
	mai = MatrixXd::Zero(m, outputlen);
	mf = MatrixXd::Zero(m, outputlen);
	maf = MatrixXd::Zero(m, outputlen);
	mc = MatrixXd::Zero(m, outputlen);
	mac = MatrixXd::Zero(m, outputlen);
	mgac = MatrixXd::Zero(m, outputlen);
	mo = MatrixXd::Zero(m, outputlen);
	mao = MatrixXd::Zero(m, outputlen);
	mgc = MatrixXd::Zero(m, outputlen);
	mh = MatrixXd::Zero(m, outputlen);
	for (int t = 0; t < m; t++)
	{
		if (t != 0)
		{
			prev_mh = mh.row(t - 1);
			prev_mc = mc.row(t - 1);
		}
		//i(t)
		mai.row(t) = (*Xs).row(t)*wix.transpose()
			+ prev_mh*wih.transpose()
			+ (prev_mc.array()*wic.transpose().array()).matrix();
		ActiFunc(&mai, t, &mi, t, gateActiFunc);
		//f(t)
		maf.row(t) = (*Xs).row(t)*wfx.transpose()
			+ prev_mh*wfh.transpose()
			+ (prev_mc.array()*wfc.transpose().array()).matrix();
		ActiFunc(&maf, t, &mf, t, gateActiFunc);
		//c(t)
		mac.row(t) = (*Xs).row(t)*wcx.transpose()
			+ prev_mh*wch.transpose();
		ActiFunc(&mac, t, &mgac, t, statActiFunc);
		mc.row(t) = mf.row(t).array()*prev_mc.array() 
			+ mi.row(t).array()*mgac.row(t).array();
		//o(t)
		mao.row(t) = (*Xs).row(t)*wox.transpose()
			+ prev_mh*woh.transpose()
			+ (mc.row(t).array()*woc.transpose().array()).matrix();
		ActiFunc(&mao, t, &mo, t, gateActiFunc);
		//j(t)
		ActiFunc(&mc, t, &mgc, t, statActiFunc);
		mh.row(t) = mo.row(t).array()*mgc.row(t).array();
	}
}
void Net::BPTT(MatrixXd &e)
{
	int m = e.rows();
	int n = e.cols();
	d_mi = MatrixXd::Zero(m + 1, outputlen);
	d_mai = MatrixXd::Zero(m + 1, outputlen);
	d_mf = MatrixXd::Zero(m + 1, outputlen);
	d_maf = MatrixXd::Zero(m + 1, outputlen);
	d_mc = MatrixXd::Zero(m + 1, outputlen);
	d_mac = MatrixXd::Zero(m + 1, outputlen);
	d_mo = MatrixXd::Zero(m + 1, outputlen);
	d_mao = MatrixXd::Zero(m + 1, outputlen);
	d_mh = MatrixXd::Zero(m + 1, outputlen);
	d_x = MatrixXd::Zero(m + 1, inputlen + 1);
	MatrixXd temp(1, outputlen);
	for (int t = m - 1; t >= 0; t--)
	{
		d_mh.row(t) = -e.row(t)
			+ d_mai.row(t + 1)*wih
			+ d_maf.row(t + 1)*wfh
			+ d_mao.row(t + 1)*woh
			+ d_mac.row(t + 1)*wch;
		
		d_mo.row(t) = d_mh.row(t).array()*mgc.row(t).array();
		ActiFunc_d(&mo, t, &temp, 0, gateActiFunc);
		d_mao.row(t) = d_mo.row(t).array()*temp.row(0).array();

		ActiFunc_d(&mgc, t, &temp, 0, statActiFunc);
		d_mc.row(t) = 
			d_mh.row(t).array()*mo.row(t).array()*temp.row(0).array()
			+ d_mai.row(t + 1).array()*wic.transpose().array()
			+ d_maf.row(t + 1).array()*wfc.transpose().array()
			+ d_mao.row(t).array()*woc.transpose().array();
		if (t < m - 1)
			d_mc.row(t) = d_mc.row(t) + (d_mc.row(t + 1).array()*mf.row(t + 1).array()).matrix();

		ActiFunc_d(&mgac, t, &temp, 0, statActiFunc);
		d_mac.row(t) = d_mc.row(t).array()
			*temp.row(0).array()
			*mi.row(t).array();

		if (t > 0)
		{
			d_mf.row(t) = d_mc.row(t).array()*mc.row(t - 1).array();
			ActiFunc_d(&mf, t, &temp, 0, gateActiFunc);
			d_maf.row(t) = d_mf.row(t).array()*temp.row(0).array();
		}

		d_mi.row(t)= d_mc.row(t).array()*mgac.row(t).array();
		ActiFunc_d(&mi, t, &temp, 0, gateActiFunc);
		d_mai.row(t) = d_mi.row(t).array()*temp.row(0).array();

		d_x.row(t) = d_mai.row(t)*wix
			+ d_maf.row(t)*wfx
			+ d_mao.row(t)*wox
			+ d_mac.row(t)*wcx;
	}

	this_d_wix = d_mai.block(0, 0, m, outputlen).transpose()*(*Xs) / m;
	this_d_wih = d_mai.block(1, 0, m - 1, outputlen).transpose()
		*mh.block(0, 0, m - 1, outputlen) / (m - 1);

	this_d_wic = (MatrixXd::Ones(1, m - 1)
		*d_mai.block(1, 0, m - 1, outputlen)
		.cwiseProduct(mc.block(0, 0, m - 1, outputlen))
		/ (m - 1)).transpose();


	this_d_wfx = d_maf.block(0, 0, m, outputlen).transpose()*(*Xs) / m;
	this_d_wfh = d_maf.block(1, 0, m - 1, outputlen).transpose()
		*mh.block(0, 0, m - 1, outputlen) / (m - 1);

	this_d_wfc = (MatrixXd::Ones(1, m - 1)
		*d_maf.block(1, 0, m - 1, outputlen)
		.cwiseProduct(mc.block(0, 0, m - 1, outputlen))
		/ (m - 1)).transpose();


	this_d_wcx = d_mac.block(0, 0, m, outputlen).transpose()*(*Xs) / m;
	this_d_wch = d_mac.block(1, 0, m - 1, outputlen).transpose()
		* mh.block(0, 0, m - 1, outputlen) / (m - 1);


	this_d_wox = d_mao.block(0, 0, m, outputlen).transpose()*(*Xs) / m;
	this_d_woh = d_mao.block(1, 0, m - 1, outputlen).transpose()
		* mh.block(0, 0, m - 1, outputlen) / (m - 1);
	this_d_woc = (MatrixXd::Ones(1, m)
		* (d_mao.block(0, 0, m, outputlen).array()
			*mc.array()).matrix()
		/ (m - 1)).transpose();

	d_wix+=this_d_wix;
	d_wih+=this_d_wih;
	d_wic+=this_d_wic;

	d_wfx+=this_d_wfx;
	d_wfh+=this_d_wfh;
	d_wfc+=this_d_wfc;

	d_wcx+=this_d_wcx;
	d_wch+=this_d_wch;

	d_wox+=this_d_wox;
	d_woh+=this_d_woh;
	d_woc+=this_d_woc;
}

void Net::Update(int thisBatchSize)
{
	if (l2_reg < 0)
		l2_reg = 0;
	//won't penalize b, only w.
	MatrixXd penalty = l2_reg*wix;
	//penalty.col(0) = MatrixXd::Zero(wix.rows(), 1);
	d_wix = -learningRate*(d_wix + penalty);
	d_wih = -learningRate*(d_wih + l2_reg*wih);
	d_wic = -learningRate*(d_wic + l2_reg*wic);
	
	penalty = l2_reg*wfx;
	//penalty.col(0) = MatrixXd::Zero(wfx.rows(), 1);
	d_wfx = -learningRate*(d_wfx + penalty);
	d_wfh = -learningRate*(d_wfh + l2_reg*wfh);
	d_wfc = -learningRate*(d_wfc + l2_reg*wfc);

	penalty = l2_reg*wcx;
	//penalty.col(0) = MatrixXd::Zero(wcx.rows(), 1);
	d_wcx = -learningRate*(d_wcx + penalty);
	d_wch = -learningRate*(d_wch + l2_reg*wch);

	penalty = l2_reg*wox;
	//penalty.col(0) = MatrixXd::Zero(wox.rows(), 1);
	d_wox = -learningRate*(d_wox + penalty);
	d_woh = -learningRate*(d_woh + l2_reg*woh);
	d_woc = -learningRate*(d_woc + l2_reg*woc);

	if (momentum < 0)
		momentum = 0;
	d_wix = (vd_wix = momentum* vd_wix + d_wix);
	d_wih = (vd_wih = momentum* vd_wih + d_wih);
	d_wic = (vd_wic = momentum* vd_wic + d_wic);

	d_wfx = (vd_wfx = momentum* vd_wfx + d_wfx);
	d_wfh = (vd_wfh = momentum* vd_wfh + d_wfh);
	d_wfc = (vd_wfc = momentum* vd_wfc + d_wfc);

	d_wcx = (vd_wcx = momentum* vd_wcx + d_wcx);
	d_wch = (vd_wch = momentum* vd_wch + d_wch);

	d_wox = (vd_wox = momentum* vd_wox + d_wox);
	d_woh = (vd_woh = momentum* vd_woh + d_woh);
	d_woc = (vd_woc = momentum* vd_woc + d_woc);

	wix = wix + d_wix / thisBatchSize;
	wih = wih + d_wih / thisBatchSize;
	wic = wic + d_wic / thisBatchSize;

	wfx = wfx + d_wfx / thisBatchSize;
	wfh = wfh + d_wfh / thisBatchSize;
	wfc = wfc + d_wfc / thisBatchSize;

	wcx = wcx + d_wcx / thisBatchSize;
	wch = wch + d_wch / thisBatchSize;

	wox = wox + d_wox / thisBatchSize;
	woh = woh + d_woh / thisBatchSize;
	woc = woc + d_woc / thisBatchSize;

	//reset d after each minibatch;
	d_wix = MatrixXd::Zero(outputlen, inputlen + 1);
	d_wih = MatrixXd::Zero(outputlen, outputlen);
	d_wic = MatrixXd::Zero(outputlen, 1);

	d_wfx = MatrixXd::Zero(outputlen, inputlen + 1);
	d_wfh = MatrixXd::Zero(outputlen, outputlen);
	d_wfc = MatrixXd::Zero(outputlen, 1);

	d_wcx = MatrixXd::Zero(outputlen, inputlen + 1);
	d_wch = MatrixXd::Zero(outputlen, outputlen);

	d_wox = MatrixXd::Zero(outputlen, inputlen + 1);
	d_woh = MatrixXd::Zero(outputlen, outputlen);
	d_woc = MatrixXd::Zero(outputlen, 1);
}
void Net::UpdateEmbedding(MatrixXd *XIds, vector<vector<double> > *embedding)
{
	int m = (*XIds).rows();
	int n = (*embedding)[0].size();
	for (int mi = 0; mi < m; mi++)//m words
	{
		for (int j = 0; j < n; j++)
		{
			(*embedding)[(*XIds)(mi, 0)][j] -= learningRate*d_x(mi, j);//0£ºword,1:pos...
		}
	}
	d_x.setZero();
}