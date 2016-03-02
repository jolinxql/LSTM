#include "Net.h"
void Cell::Setup(int _inputlen
	, int _outputlen
	, int _hiddenUnits
	, string _gateActiFunc
	, string _statActiFunc
	, double  _weightRange
	, double _learningRate
	, double _l2_reg
	, double _momentum)
{
	inputlen = _inputlen;
	outputlen = _outputlen;
	hiddenUnits = _hiddenUnits;
	gateActiFunc = _gateActiFunc;
	statActiFunc = _statActiFunc;
	weightRange = _weightRange;
	learningRate = _learningRate;
	l2_reg = _l2_reg;
	momentum = _momentum;

	wix = MatrixXd::Random(outputlen, inputlen + 1) / 2 / inputlen * weightRange;
	wih = MatrixXd::Random(outputlen, hiddenUnits) / 2 / outputlen * weightRange;
	wic = MatrixXd::Random(outputlen, 1) / 2 / outputlen * weightRange;

	wfx = MatrixXd::Random(outputlen, inputlen + 1) / 2 / inputlen * weightRange;
	wfh = MatrixXd::Random(outputlen, hiddenUnits) / 2 / outputlen * weightRange;
	wfc = MatrixXd::Random(outputlen, 1) / 2 / outputlen * weightRange;

	wcx = MatrixXd::Random(outputlen, inputlen + 1) / 2 / inputlen * weightRange;
	wch = MatrixXd::Random(outputlen, hiddenUnits) / 2 / outputlen * weightRange;

	wox = MatrixXd::Random(outputlen, inputlen + 1) / 2 / inputlen * weightRange;
	woh = MatrixXd::Random(outputlen, hiddenUnits) / 2 / outputlen * weightRange;
	woc = MatrixXd::Random(outputlen, 1) / 2 / outputlen * weightRange;

	vd_wix = MatrixXd::Zero(outputlen, inputlen + 1);
	vd_wih = MatrixXd::Zero(outputlen, hiddenUnits);
	vd_wic = MatrixXd::Zero(outputlen, 1);

	vd_wfx = MatrixXd::Zero(outputlen, inputlen + 1);
	vd_wfh = MatrixXd::Zero(outputlen, hiddenUnits);
	vd_wfc = MatrixXd::Zero(outputlen, 1);

	vd_wcx = MatrixXd::Zero(outputlen, inputlen + 1);
	vd_wch = MatrixXd::Zero(outputlen, hiddenUnits);

	vd_wox = MatrixXd::Zero(outputlen, inputlen + 1);
	vd_woh = MatrixXd::Zero(outputlen, hiddenUnits);
	vd_woc = MatrixXd::Zero(outputlen, 1);

	d_wix = MatrixXd::Zero(outputlen, inputlen + 1);
	d_wih = MatrixXd::Zero(outputlen, hiddenUnits);
	d_wic = MatrixXd::Zero(outputlen, 1);

	d_wfx = MatrixXd::Zero(outputlen, inputlen + 1);
	d_wfh = MatrixXd::Zero(outputlen, hiddenUnits);
	d_wfc = MatrixXd::Zero(outputlen, 1);

	d_wcx = MatrixXd::Zero(outputlen, inputlen + 1);
	d_wch = MatrixXd::Zero(outputlen, hiddenUnits);

	d_wox = MatrixXd::Zero(outputlen, inputlen + 1);
	d_woh = MatrixXd::Zero(outputlen, hiddenUnits);
	d_woc = MatrixXd::Zero(outputlen, 1);
}
void Cell::FF(int t, MatrixXd prev_mh)
{
	int m = (*Xs).rows();
	int n = (*Xs).cols();//n=inputlen + 1
	if (t == 0)
	{
		prev_mc = MatrixXd::Zero(1, outputlen);

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
	}
	if (t != 0)
	{
		prev_mc = mc.row(t - 1);
	}
	//i(t)
	mai.row(t) = (*Xs).row(t)*wix.transpose()
		+ prev_mh.cwiseProduct(wih.transpose()).colwise().sum().matrix()
		+ (prev_mc.array()*wic.transpose().array()).matrix();
	ActiFunc(&mai, t, &mi, t, gateActiFunc);
	//f(t)
	maf.row(t) = (*Xs).row(t)*wfx.transpose()
		+ prev_mh.cwiseProduct(wfh.transpose()).colwise().sum().matrix()
		+ (prev_mc.array()*wfc.transpose().array()).matrix();
	ActiFunc(&maf, t, &mf, t, gateActiFunc);
	//c(t)
	mac.row(t) = (*Xs).row(t)*wcx.transpose()
		+ prev_mh.cwiseProduct(wch.transpose()).colwise().sum().matrix();
	ActiFunc(&mac, t, &mgac, t, statActiFunc);
	mc.row(t) = mf.row(t).array()*prev_mc.array()
		+ mi.row(t).array()*mgac.row(t).array();
	//o(t)
	mao.row(t) = (*Xs).row(t)*wox.transpose()
		+ prev_mh.cwiseProduct(woh.transpose()).colwise().sum().matrix()
		+ (mc.row(t).array()*woc.transpose().array()).matrix();
	ActiFunc(&mao, t, &mo, t, gateActiFunc);
	//j(t)
	ActiFunc(&mc, t, &mgc, t, statActiFunc);
	mh.row(t) = mo.row(t).array()*mgc.row(t).array();
}
void Cell::BPTT(int t)
{
	MatrixXd temp(1, outputlen);

	d_mo.row(t) = d_mh.row(t).array()*mgc.row(t).array();
	ActiFunc_d(&mo, t, &temp, 0, gateActiFunc);
	d_mao.row(t) = d_mo.row(t).array()*temp.row(0).array();

	ActiFunc_d(&mgc, t, &temp, 0, statActiFunc);
	d_mc.row(t) =
		d_mh.row(t).array()*mo.row(t).array()*temp.row(0).array()
		+ d_mai.row(t + 1).array()*wic.transpose().array()
		+ d_maf.row(t + 1).array()*wfc.transpose().array()
		+ d_mao.row(t).array()*woc.transpose().array();
	if (t < mf.rows() - 1)
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

	d_mi.row(t) = d_mc.row(t).array()*mgac.row(t).array();
	ActiFunc_d(&mi, t, &temp, 0, gateActiFunc);
	d_mai.row(t) = d_mi.row(t).array()*temp.row(0).array();

	d_x.row(t) = d_mai.row(t)*wix
		+ d_maf.row(t)*wfx
		+ d_mao.row(t)*wox
		+ d_mac.row(t)*wcx;
}

void Cell::BPTT2(vector<MatrixXd> *all_mh)
{
	int m = d_mh.rows() - 1;
	this_d_wix = d_mai.block(0, 0, m, outputlen).transpose()*(*Xs) / m;
	
	this_d_wih = MatrixXd::Zero(wih.rows(),wih.cols());
	for (int t = 1; t < m; t++)
		for (int h = 0; h < hiddenUnits; h++)
			this_d_wih.col(h) += d_mai.row(t).cwiseProduct((*all_mh)[h].row(t - 1)).transpose();
	this_d_wih = this_d_wih / (m - 1);

	this_d_wic = (MatrixXd::Ones(1, m - 1)
		*d_mai.block(1, 0, m - 1, outputlen)
		.cwiseProduct(mc.block(0, 0, m - 1, outputlen))
		/ (m - 1)).transpose();


	this_d_wfx = d_maf.block(0, 0, m, outputlen).transpose()*(*Xs) / m;
	this_d_wfh = MatrixXd::Zero(wfh.rows(), wfh.cols());
	for (int t = 1; t < m; t++)
		for (int h = 0; h < hiddenUnits; h++)
			this_d_wfh.col(h) += d_maf.row(t).cwiseProduct((*all_mh)[h].row(t - 1)).transpose();
	this_d_wfh = this_d_wfh / (m - 1);

	this_d_wfc = (MatrixXd::Ones(1, m - 1)
		*d_maf.block(1, 0, m - 1, outputlen)
		.cwiseProduct(mc.block(0, 0, m - 1, outputlen))
		/ (m - 1)).transpose();


	this_d_wcx = d_mac.block(0, 0, m, outputlen).transpose()*(*Xs) / m;
	this_d_wch = MatrixXd::Zero(wch.rows(), wch.cols());
	for (int t = 1; t < m; t++)
		for (int h = 0; h < hiddenUnits; h++)
			this_d_wch.col(h) += d_mac.row(t).cwiseProduct((*all_mh)[h].row(t - 1)).transpose();
	this_d_wch = this_d_wch / (m - 1);


	this_d_wox = d_mao.block(0, 0, m, outputlen).transpose()*(*Xs) / m;
	this_d_woh = MatrixXd::Zero(woh.rows(), woh.cols());
	for (int t = 1; t < m; t++)
		for (int h = 0; h < hiddenUnits; h++)
			this_d_woh.col(h) += d_mao.row(t).cwiseProduct((*all_mh)[h].row(t - 1)).transpose();
	this_d_woh = this_d_woh / (m - 1);
	this_d_woc = (MatrixXd::Ones(1, m)
		* (d_mao.block(0, 0, m, outputlen).array()
			*mc.array()).matrix()
		/ (m - 1)).transpose();

	d_wix += this_d_wix;
	d_wih += this_d_wih;
	d_wic += this_d_wic;

	d_wfx += this_d_wfx;
	d_wfh += this_d_wfh;
	d_wfc += this_d_wfc;

	d_wcx += this_d_wcx;
	d_wch += this_d_wch;

	d_wox += this_d_wox;
	d_woh += this_d_woh;
	d_woc += this_d_woc;
}

void Cell::ActiFunc(MatrixXd *x, int xt, MatrixXd *y, int yt, string ActiFunc)
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
}

void Cell::ActiFunc_d(MatrixXd *x, int xt, MatrixXd *y, int yt, string ActiFunc)
{
	if (strcmp("sigm", ActiFunc.c_str()) == 0)
	{
		(*y).row(yt) = (*x).row(xt).array()*(1 - (*x).row(xt).array());
	}
	else if (strcmp("tanh_opt", ActiFunc.c_str()) == 0)
	{
		(*y) = 1.7159 * 2 / 3 * (1 - 1 / (1.7159*1.7159)*(*x).array()*(*x).array());
	}
	else if (strcmp("linear", ActiFunc.c_str()) == 0)
	{
		(*y) = MatrixXd::Ones((*x).rows(), (*x).cols());
	}
}
void Cell::Update(int thisBatchSize)
{
	//MatrixXd penalty = l2_reg*wix;
	//penalty.col(0) = MatrixXd::Zero(wix.rows(), 1);
	d_wix = -learningRate*(d_wix + l2_reg*wix);
	d_wih = -learningRate*(d_wih + l2_reg*wih);
	d_wic = -learningRate*(d_wic + l2_reg*wic);

	//penalty = l2_reg*wfx;
	//penalty.col(0) = MatrixXd::Zero(wfx.rows(), 1);
	d_wfx = -learningRate*(d_wfx + l2_reg*wfx);
	d_wfh = -learningRate*(d_wfh + l2_reg*wfh);
	d_wfc = -learningRate*(d_wfc + l2_reg*wfc);

	//penalty = l2_reg*wcx;
	//penalty.col(0) = MatrixXd::Zero(wcx.rows(), 1);
	d_wcx = -learningRate*(d_wcx + l2_reg*wcx);
	d_wch = -learningRate*(d_wch + l2_reg*wch);

	//penalty = l2_reg*wox;
	//penalty.col(0) = MatrixXd::Zero(wox.rows(), 1);
	d_wox = -learningRate*(d_wox + l2_reg*wox);
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
	d_wix.setZero();
	d_wih.setZero();
	d_wic.setZero();

	d_wfx.setZero();
	d_wfh.setZero();
	d_wfc.setZero();

	d_wcx.setZero();
	d_wch.setZero();

	d_wox.setZero();
	d_woh.setZero();
	d_woc.setZero();
}



void Net::mycoutmat(MatrixXd * out)
{
	cout << (*out) << endl;
	getchar();
}
void Net::Setup(int _inputlen
	, int _outputlen
	, int _hiddenUnits
	, string _gateActiFunc
	, string _statActiFunc
	, double  _weightRange
	, double _learningRate
	, double _l2_reg
	, double _momentum)
{
	inputlen = _inputlen;
	outputlen = _outputlen;
	hiddenUnits = _hiddenUnits;
	gateActiFunc = _gateActiFunc;
	statActiFunc = _statActiFunc;
	weightRange = _weightRange;
	learningRate = _learningRate;
	l2_reg = _l2_reg;
	momentum = _momentum;

	wkh = MatrixXd::Random(hiddenUnits, outputlen) / 2 / hiddenUnits * weightRange;
	for (int i = 0; i < hiddenUnits; i++)
	{
		Cell cell;
		cell.Setup(inputlen, outputlen, hiddenUnits, gateActiFunc, statActiFunc, weightRange, learningRate, l2_reg, momentum);
		hiddenCells.push_back(cell);
	}
	d_wkh = MatrixXd::Zero(hiddenUnits, outputlen);
	vd_wkh = MatrixXd::Zero(hiddenUnits, outputlen);
	this_d_wkh = MatrixXd::Zero(hiddenUnits, outputlen);
}


void Net::Softmax(MatrixXd *x, int xt, MatrixXd *y, int yt)
{
	double sum = ((*x).row(xt)).array().exp().sum();
	(*y).row(yt) = (*x).row(xt).array().exp() / sum;
}
void Net::FF()
{
	int m = (*Xs).rows();
	MatrixXd prev_bh = MatrixXd::Zero(hiddenUnits, outputlen);
	MatrixXd temp = MatrixXd::Zero(hiddenUnits, outputlen);

	all_mh.clear();
	for (int h = 0; h < hiddenUnits; h++)
	{
		all_mh.push_back(MatrixXd::Zero(m, outputlen));
	}
	for (int t = 0; t < m; t++)
	{
		temp = MatrixXd::Zero(hiddenUnits, outputlen);
		for (int h = 0; h < hiddenUnits; h++)
		{
			hiddenCells[h].Xs = Xs;
			hiddenCells[h].FF(t, prev_bh);
			temp.row(h) = hiddenCells[h].mh.row(t).array();
			all_mh[h].row(t) = hiddenCells[h].mh.row(t).array();
		}
		prev_bh = temp;
	}

	mak = MatrixXd::Zero(m, outputlen);
	for (int h = 0; h < hiddenUnits; h++)
	{
		for (int j = 0; j < outputlen; j++)
			mak.col(j) += wkh(h, j) * hiddenCells[h].mh.col(j);
	}
	myk = MatrixXd::Zero(m, outputlen);
	for (int t = 0; t < m; t++)
	{
		Softmax(&mak, t, &myk, t);
	}
}
double Net::Loss()
{
	e = (*Ys) - myk;
	double loss=-((*Ys).array()*myk.array().log()).sum() / (*Xs).rows();
	return loss;
}
void Net::BPTT()
{
	int m = (*Xs).rows();

	//d_mak = -e;
	this_d_wkh.setZero();
	for (int h = 0; h < hiddenUnits; h++)
	{
		for (int j = 0; j < outputlen; j++)
			this_d_wkh(h, j) += (-e.col(j).array() * hiddenCells[h].mh.col(j).array()).sum()/m;
	}
	d_wkh += this_d_wkh;
	for (int h = 0; h < hiddenUnits; h++)
	{
		hiddenCells[h].d_mh = MatrixXd::Zero(m + 1, outputlen);
		hiddenCells[h].d_mi = MatrixXd::Zero(m + 1, outputlen);
		hiddenCells[h].d_mai = MatrixXd::Zero(m + 1, outputlen);
		hiddenCells[h].d_mf = MatrixXd::Zero(m + 1, outputlen);
		hiddenCells[h].d_maf = MatrixXd::Zero(m + 1, outputlen);
		hiddenCells[h].d_mc = MatrixXd::Zero(m + 1, outputlen);
		hiddenCells[h].d_mac = MatrixXd::Zero(m + 1, outputlen);
		hiddenCells[h].d_mo = MatrixXd::Zero(m + 1, outputlen);
		hiddenCells[h].d_mao = MatrixXd::Zero(m + 1, outputlen);
		hiddenCells[h].d_x = MatrixXd::Zero(m + 1, inputlen + 1);
		for (int j = 0; j < outputlen; j++)
			hiddenCells[h].d_mh.block(0,j,m,1) = -e.col(j)*wkh(h, j);
	}
	for (int t = m - 1; t >= 0; t--)
	{
		MatrixXd temp = MatrixXd::Zero(1, outputlen);
		for (int h = 0; h < hiddenUnits; h++)
		{
			temp += (hiddenCells[h].d_mai.row(t + 1).array()*hiddenCells[h].wih.transpose().row(h).array()).matrix();
		}
		for (int h = 0; h < hiddenUnits; h++)
		{
			hiddenCells[h].d_mh.row(t) += temp;
			hiddenCells[h].BPTT(t);
		}
	}
	for (int h = 0; h < hiddenUnits; h++)
	{
		hiddenCells[h].BPTT2(&all_mh);
	}
	d_x = MatrixXd::Zero(m, inputlen);
	for (int h = 0; h < hiddenUnits; h++)
	{
		d_x += hiddenCells[h].d_x.block(0, 0, m, inputlen);
	}
}
void Net::Update(int thisBatchSize)
{
	if (l2_reg < 0)
		l2_reg = 0;
	d_wkh = -learningRate*(d_wkh + l2_reg*wkh);
	if (momentum < 0)
		momentum = 0;
	d_wkh = (vd_wkh = momentum* vd_wkh + d_wkh);
	wkh = wkh + d_wkh / thisBatchSize;

	d_wkh.setZero();

	for (int h = 0; h < hiddenUnits; h++)
	{
		hiddenCells[h].Update(thisBatchSize);
	}

}
void Net::UpdateEmbedding(MatrixXd *XIds, vector<vector<double> > *embedding)
{
	int m = (*XIds).rows();
	for (int mi = 0; mi < m; mi++)
	{
		for (int j = 0; j < inputlen; j++)
			(*embedding)[(*XIds)(mi, 0)][j] -= learningRate*d_x(mi, j);//0£ºword,1:pos...
	}
}