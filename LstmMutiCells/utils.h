#ifndef UTIL_H___
#define UTIL_H___

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <Eigen/Dense>
#include <algorithm>
#include "Net.h"
#include <time.h>
#include <iostream>
using Eigen::MatrixXd;
//using Eigen::ArrayXXd;

using namespace std;
struct Config
{
	int totalCount;
	int batchSize;
	int epochNum;
	int useBatchAdapt;
	int len_in;
	int len_out;
	int hiddenUnits;
	string gateActiFunc;
	string statActiFunc;
	string outActiFunc;
	double weightRange;
	double learningRate;	//alpha
	double l2_reg;			//weightPenalty;
	double momentum;
	//double scalingLearningRate;
};
void Split(const string &input, const char spliter, vector<string> *out);
int ReadConfig(const string &config_file, Config* config);

int ReadWordEmbedding(const string &file_name,
	map<string, int> *word_dict,
	vector<string>* id_to_word,
	vector<vector<double> > *word_embedding,
	int count=-1);

// read the dependency data from file
struct SeqNode
{
	int word;
	int pos;
	/*int type;
	int head;
	bool punc;*/
};

void AddSpecialPOS(const string &POS,
	map<string, int>* pos_dict,
	vector<string>* id_to_pos);
bool IsChinese(const string &input);
bool IsDigit(const string &input, const string &POS, bool chinese);
int ReadDepData(const string &file_name,
	map<string, int> &word_dict,
	map<string, int> *pos_dict,
	vector<string> *id_to_pos,
	//map<string, int> *edge_type_dict,
	//vector<string> *id_to_edge_type,
	vector<vector<SeqNode> > *out);


void GradientCheckFull(Net &net);
void GradientCheck(Net *net, MatrixXd *Ys, MatrixXd &w, MatrixXd &this_d_w);
void CopyToMatrix(const vector<vector<SeqNode> >& input,
	vector<vector<double> > &word_embedding,
	int inputlen,
	int outputlen,
	vector<vector<MatrixXd> > *XYs,
	vector<MatrixXd> *XsIds);
void Train(vector<vector<SeqNode> > train_data,
	vector<vector<double> > word_embedding,
	Config config, Net &net);
void Test(vector<vector<SeqNode> > test_data,
	vector<vector<double> > word_embedding, Net &net);
#endif