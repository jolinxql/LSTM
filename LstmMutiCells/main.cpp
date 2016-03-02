#include "utils.h"

using namespace std;
int main(int argc, char* argv[])
{
#ifndef _WIN32
	//linux
	if (argc != 5)
	{
		printf("Usage: ./Lstm.exe train_data dev_data word_embedding config_file\n");
		return 1;
	}
	//�����updateEmbedding �Լ� ReadAMRData()��

	//�Ѿ����˵ģ�embedding update,useBatchAdapt,momentum,gradient check
	//�л��᣺
	//dropout,
	//saveparam,
	//save sentence embedding,
	//�ҵ����������ĵ��Լ�������
	//���⼸��������Ĳ���
	//curBatchSize����
	//trick�����ԭʼ��������ѵ��������1000�Σ�����minibatch����Ҫ��֤update�ĸ��´���������1000��
	//linux�µ�����Ҫ��UNIX��ʽ����notepad++�����
#else
	//windows
	argv[1] = "../train_data.txt";
	argv[2] = "../dev_data.txt";
	argv[3] = "../word_vectors_50.txt";
	argv[4] = "config.txt";
#endif
	Config config;

	map<string, int> word_dict;
	vector<string> id_to_word;
	map<string, int> pos_dict;
	vector<string> id_to_pos;
	vector<vector<double> > word_embedding;

	vector<vector<SeqNode> > train_data;
	vector<vector<SeqNode> > dev_data;

	if(-1==ReadConfig(argv[4], &config))
	{
		printf("Reading config file fail!!!");
		getchar(); return -1;
	}

	printf("Reading word embedding...\n");
	if (-1==ReadWordEmbedding(argv[3], &word_dict, 
		&id_to_word, &word_embedding))
	{
		printf("Reading word embedding fail!!!");
		getchar(); return -1;
	}
	printf("\tembedding: %d\n",word_embedding.size());

	printf("Reading training data...\n");
	if (-1 == ReadDepData(argv[1], word_dict,
		&pos_dict, &id_to_pos, &train_data))
	{
		printf("Reading training data fail!!!");
		getchar(); return -1;
	}
	printf("\t%d\n", train_data.size());
	printf("\tpos_dict: %d\n", pos_dict.size());
	printf("\tid_to_pos: %d\n", id_to_pos.size());

	printf("Reading dev data...\n");
	if (-1 == ReadDepData(argv[2], word_dict,
		&pos_dict, &id_to_pos, &dev_data))
	{
		printf("Reading dev data fail!!!");
		getchar(); return -1;
	}
	printf("\t%d\n", dev_data.size());
	printf("\tpos_dict: %d\n", pos_dict.size());
	printf("\tid_to_pos: %d\n", id_to_pos.size());
	
	config.totalCount = train_data.size();
	config.len_in = word_embedding[0].size();
	config.len_out = pos_dict.size();

	Net net;
	Train(train_data, word_embedding, config, net);
	
	Test(dev_data, word_embedding, net);
	getchar();
	return 0;
}