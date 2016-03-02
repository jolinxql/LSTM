#include "utils.h"
void Split(const string &input, const char spliter, vector<string> *out)
{
	int len = input.size();

	if (len == 0)
		return;
	if (out == NULL)
		return;

	string tmp;
	for (int i = 0; i < len; ++i)
	{
		if (input[i] != spliter)
		{
			// Append one char to the end of the string
			tmp.append(1, input[i]);
		}
		else
		{
			out->push_back(tmp);
			tmp.clear();
		}
	}

	if (tmp.size() != 0)
		out->push_back(tmp);
}
int ReadConfig(const string &config_file, Config* config)
{
	ifstream fin(config_file.c_str());
	map<string, string> dict;

	if (fin.is_open())
	{
		string buf;

		while (getline(fin, buf))
		{
			//buf = buf.substr(0, buf.size() - 1);
			vector<string> field;

			Split(buf, '=', &field);
			dict[field[0]] = field[1];
		}
		fin.close();
		config->hiddenUnits= atoi(dict["hiddenUnits"].c_str());
		config->totalCount = atoi(dict["totalCount"].c_str());
		config->batchSize = atoi(dict["batchSize"].c_str());
		config->epochNum = atoi(dict["epochNum"].c_str());
		config->useBatchAdapt = atoi(dict["useBatchAdapt"].c_str());
		config->len_in = atoi(dict["len_in"].c_str());
		config->len_out = atoi(dict["len_out"].c_str());
		config->weightRange = atof(dict["weightRange"].c_str());
		config->learningRate = atof(dict["learningRate"].c_str());
		config->l2_reg = atof(dict["l2_reg"].c_str());
		config->momentum = atof(dict["momentum"].c_str());
		config->gateActiFunc = dict["gateActiFunc"];
		config->statActiFunc = dict["statActiFunc"];
		config->outActiFunc= dict["outActiFunc"];
	}
	else
	{
		return -1;
	}
	return 0;

}
int ReadWordEmbedding(const string &file_name,
	map<string, int> *word_dict,
	vector<string>* id_to_word,
	vector<vector<double> > *word_embedding,
	int count)
{
	ifstream fin(file_name.c_str());

	if (fin.is_open())
	{
		string buf;

		while (getline(fin, buf))
		{
			vector<double> vec;
			vector<string> field;

			// get rid of \n
			if (buf.size() != 0 && (buf[buf.size() - 1] == '\n' || buf[buf.size() - 1] == '\r'))
			{
				buf = buf.substr(0, buf.size() - 1);
			}

			Split(buf, ' ', &field);
			if (word_dict->find(field[0]) == word_dict->end())
			{
				int id = word_dict->size();
				(*word_dict)[field[0]] = id;
				id_to_word->push_back(field[0]);
			}
			int embed_size = field.size() - 1;

			for (int i = 0; i < embed_size; i++)
				vec.push_back(atof(field[i + 1].c_str()));
			word_embedding->push_back(vec);
			if (count != -1)
				if (word_embedding->size() > count)
					break;
		}
	}
	else
		return -1;
	return 0;
}

#pragma region ReadDepData============================================================
void AddSpecialPOS(const string &POS,
	map<string, int>* pos_dict,
	vector<string>* id_to_pos)
{
	int cur_size = pos_dict->size();
	(*pos_dict)[POS] = cur_size;
	id_to_pos->push_back(POS);
}
bool IsChinese(const string &input)
{
	int len = input.size();

	for (int i = 0; i < len; i++)
	{
		if (input[i] >= 'a' && input[i] <= 'z')
			return false;
		if (input[i] >= 'A' && input[i] <= 'Z')
			return false;
	}

	return true;
}
// Judge whether a string is a number
// number examples: 100; 100,000; 1/2; 1.25

bool IsDigit(const string &input, const string &POS, bool chinese)
{
	int len = input.size();
	bool digit_appear = false;

	if (len == 0)
		return false;

	if (chinese && POS == "CD")
		return true;

	for (int i = 0; i < len; i++)
	{
		if (input[i] >= '0' && input[i] <= '9')
			digit_appear = true;
		else if (input[i] == '.' || input[i] == ',' || input[i] == '-' || input[i] == '/')
			continue;
		else
			return false;
	}

	return digit_appear;
}

bool IsPunc(const string &POS)
{
	if (POS == "." || POS == "," || POS == ":" || POS == "``" || POS == "''" || POS == "PU")
		return true;
	return false;
}
// Read the dependency data and encode them
// return the encoded data and updated pos && edge type dict
int ReadDepData(const string &file_name,
	map<string, int> &word_dict,
	map<string, int> *pos_dict,
	vector<string>* id_to_pos,
	//map<string, int> *edge_type_dict,
	//vector<string>* id_to_edge_type,
	vector<vector<SeqNode> > *out)
{
	vector<SeqNode> seq;
	SeqNode root;
	bool chinese = false;
	bool first = true;
	map<string, int>::iterator iter;
	ifstream fin(file_name.c_str());

	// First Add <BOS> and <EOS> to the POS dict
	// So <BOS> id is always 0 and <EOS> 1
	if (pos_dict->find("<BOS>") == pos_dict->end())
		AddSpecialPOS("<BOS>", pos_dict, id_to_pos);
	if (pos_dict->find("<EOS>") == pos_dict->end())
		AddSpecialPOS("<EOS>", pos_dict, id_to_pos);

	// The ROOT node
	if (pos_dict->find("<ROOT>") == pos_dict->end())
		AddSpecialPOS("<ROOT>", pos_dict, id_to_pos);
	root.word = word_dict["<ROOT>"];
	root.pos = 2; // The ID for root
	//root.punc = false; // it's not punc
	//root.head = -1; // The head for root is meaningless
	//root.type = -1; // The head for root is meaningless
	//				//Add ROOT node
	seq.push_back(root);

	if (fin.is_open())
	{
		string buf;

		while (getline(fin, buf))
		{
			if (buf.size() != 0 && (buf[buf.size() - 1] == '\n' || buf[buf.size() - 1] == '\r'))
			{
				buf = buf.substr(0, buf.size() - 1);
			}

			if (buf.size() == 0) // end of sen
			{
				out->push_back(seq);
				seq.clear();
				// Add ROOT
				seq.push_back(root);
			}
			else
			{
				vector<string> field;
				SeqNode tmp;

				Split(buf, '\t', &field);

				if (first)
				{
					chinese = IsChinese(field[0]);
					first = false;
				}
				// encode word
				string word(field[0]);
				if (IsDigit(field[0], field[1], chinese))
					word = "<NUM>";
				iter = word_dict.find(word);
				if (iter != word_dict.end())
					tmp.word = iter->second;
				else
					tmp.word = word_dict["<OOV>"];
				// encode pos
				iter = pos_dict->find(field[1]);
				if (iter != pos_dict->end())
					tmp.pos = iter->second;
				else
				{
					tmp.pos = pos_dict->size();
					(*pos_dict)[field[1]] = tmp.pos;
					id_to_pos->push_back(field[1]);
				}
				//// whether this is punc
				//tmp.punc = IsPunc(field[1]);
				//// save head
				//tmp.head = atoi(field[2].c_str());
				// encode edge type
				/*iter = edge_type_dict->find(field[3]);
				if (iter != edge_type_dict->end())
					tmp.type = iter->second;
				else
				{
					tmp.type = edge_type_dict->size();
					(*edge_type_dict)[field[3]] = tmp.type;
					id_to_edge_type->push_back(field[3]);
				}
				*/
				seq.push_back(tmp);
			}
		}

		fin.close();
	}
	else
	{
		return -1;
	}

	return 0;
}
#pragma endregion


#pragma region StartTrain============================================================
void CopyToMatrixList(const vector<vector<SeqNode> >& data,
	vector<vector<double> > &word_embedding,
	int inputlen,
	int outputlen,
	vector<vector<MatrixXd> > *XYs,
	vector<MatrixXd> *XsIds)
{
	int n = data.size();
	int embedding_size = word_embedding[0].size();

		for (int i = 0; i < n; i++)//n sentences
		{
			XYs->push_back(vector<MatrixXd>(2));
			int m = data[i].size();

			(*XYs)[i][0] = MatrixXd::Zero(m, inputlen + 1);
			(*XYs)[i][1] = MatrixXd::Zero(m, outputlen);

			if (XsIds != NULL)//only when training
				XsIds->push_back(MatrixXd::Zero(m, 1));
			for (int j = 0; j < m; j++)//m words
			{
				vector<double> embedding = word_embedding[data[i][j].word];
				if (XsIds != NULL)
				{
					(*XsIds)[i](j, 0) = data[i][j].word;
				}

				for (int k = 0; k < inputlen; k++)
					(*XYs)[i][0](j,k) = embedding[k];
				(*XYs)[i][0](j, inputlen) = 1;//weight b
				for (int k = 0; k < outputlen; k++)
					(*XYs)[i][1](j, k) = (k == data[i][j].pos) ? 1 : 0;
			}
		}
}
void GradientCheck(Net *net, MatrixXd &w, MatrixXd &this_d_w)
{
	int m = w.rows();
	int n = w.cols();
	const double eps = 1.0e-5;

	MatrixXd fabsDiff(m, n);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			w(i, j) += eps;
			//cout << w(i, j)<<"\t";
			(*net).FF();
			double loss1 = (*net).Loss();
			(*net).FF();
			double loss2 = (*net).Loss();
			double gradient = (loss1 - loss2) / (2 * eps);
			fabsDiff(i, j) = fabs(gradient - this_d_w(i, j));
			w(i, j) += eps;
		}
	}
	cout << fabsDiff.mean() << endl;
}
void GradientCheck(Cell *cell, Net *net, MatrixXd &w, MatrixXd &this_d_w)
{
	int m = w.rows();
	int n = w.cols();
	const double eps = 1.0e-5;

	MatrixXd fabsDiff(m, n);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			w(i, j) += eps;
			//cout << w(i, j)<<"\t";
			(*net).FF();
			double loss1 = (*net).Loss();
			(*net).FF();
			double loss2 = (*net).Loss();
			double gradient = (loss1 - loss2) / (2 * eps);
			fabsDiff(i, j) = fabs(gradient - this_d_w(i, j));
			w(i, j) += eps;
		}
	}
	cout << fabsDiff.mean() << endl;
}
void GradientCheckFull(Net &net)
{
	cout << "d_wkh: ";
	GradientCheck(&net, net.wkh, net.this_d_wkh);
	cout << "d_x: ";
	GradientCheck(&net, *(net.Xs), net.d_x);
	cout << "for each cell in " << net.hiddenUnits << " cells" << endl;
	for (int h = 0; h < net.hiddenUnits; h++)
	{
		cout << "d_wix: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].wix, net.hiddenCells[h].this_d_wix);
		cout << "d_wih: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].wih, net.hiddenCells[h].this_d_wih);
		cout << "d_wic: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].wic, net.hiddenCells[h].this_d_wic);
		cout << "d_wfx: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].wfx, net.hiddenCells[h].this_d_wfx);
		cout << "d_wfh: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].wfh, net.hiddenCells[h].this_d_wfh);
		cout << "d_wfc: ";	
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].wfc, net.hiddenCells[h].this_d_wfc);
		cout << "d_wcx: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].wcx, net.hiddenCells[h].this_d_wcx);
		cout << "d_wch: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].wch, net.hiddenCells[h].this_d_wch);
		cout << "d_wox: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].wox, net.hiddenCells[h].this_d_wox);
		cout << "d_woh: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].woh, net.hiddenCells[h].this_d_woh);
		cout << "d_woc: ";
		GradientCheck(&(net.hiddenCells[h]), &net, net.hiddenCells[h].woc, net.hiddenCells[h].this_d_woc);
	}

	//getchar();
}
void Train(vector<vector<SeqNode> > train_data,
	vector<vector<double> > word_embedding,
	Config config, Net &net)
{
	//trainXYs[i] = sentence i
	//trainXYs[i][0] = Xs of sent i
	//trainXYs[i][1] = Ys of sent i
	vector<vector<MatrixXd> > trainXYs;
	vector<MatrixXd> trainXsId;
	CopyToMatrixList(train_data, word_embedding,
		config.len_in, config.len_out, &trainXYs, &trainXsId);

	int train_size = train_data.size();

	net.Setup(config.len_in
		, config.len_out
		, config.hiddenUnits
		, config.gateActiFunc
		, config.statActiFunc
		, config.weightRange
		, config.learningRate
		, config.l2_reg
		, config.momentum);

	double start_time = clock();
	for (int ei = 0; ei < config.epochNum; ei++)
	{
		cout << ei;
		int curBatchSize = config.batchSize;
		if (config.useBatchAdapt > 0)
		{
			// sum(1./(1-1./exp((batchSize/train_size)+0.012*ei)))=1000时（Update次数加起来约等于1000次,epoch=550）
			curBatchSize = 0.4*train_size*(1 - 1. / exp((double)config.batchSize / (double)train_size + 0.012*ei));
			cout << " current batch size: " << curBatchSize;
		}
		int batchNum = train_size / curBatchSize + 1;

		double loss = 0.0;
		for (int bi = 0; bi < batchNum; bi++)
		{
			int startIdx = bi*curBatchSize;
			if (startIdx >= train_size)
				break;
			int endIdx = min((bi + 1)*curBatchSize, train_size);
			for (int i = startIdx; i < endIdx; i++)
			{
				net.Xs = &(trainXYs[i][0]);
				net.Ys = &(trainXYs[i][1]);
				net.FF();
				loss += net.Loss();
				net.BPTT();
				//GradientCheckFull(net);

				net.UpdateEmbedding(&(trainXsId[i]), &word_embedding);
			}
			net.Update(endIdx - startIdx);

			//update para	
		}
		loss /= train_size;
		cout << " , Loss:" << loss << endl;
	}
	cout << "time:" << (clock() - start_time) / (double)CLOCKS_PER_SEC << endl;
}
void Test(vector<vector<SeqNode> > test_data,
	vector<vector<double> > word_embedding, Net &net)
{
	vector<vector<MatrixXd> > testXYs;
	CopyToMatrixList(test_data, word_embedding, net.inputlen, net.outputlen, &testXYs, NULL);

	int total = 0, posTrue = 0;
	int test_size = test_data.size();
	for (int i = 0; i < test_size; i++)
	{
		net.Xs = &(testXYs[i][0]);
		net.Ys = &(testXYs[i][1]);
		net.FF();
		for (int mi = 0; mi < net.Xs->rows(); mi++)//words
		{
			int maxTypeIdx;
			double max = -1000000000;
			for (int ti = 0; ti < net.Ys->cols(); ti++)//types
			{
				if (net.myk(mi, ti)>max)
				{
					maxTypeIdx = ti; max = net.myk(mi, ti);
				}
			}
			if ((*(net.Ys))(mi, maxTypeIdx) == 1)
				posTrue++;
			total++;
		}
	}

	cout << posTrue << endl;
	cout << total << endl;
	cout << (double)posTrue / (double)total << endl;
}
#pragma endregion