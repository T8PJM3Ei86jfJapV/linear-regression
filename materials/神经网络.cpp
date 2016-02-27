//����λ��������תΪһλʮ������

#include<iostream>
#include<cmath>
#include<fstream>
#include<limits.h>
#include<string>

#pragma warning(disable:4996)

using namespace std;

#define  innode 384  //��������
#define  hidenode 100//���������
#define  outnode 1 //��������
#define  TRAINSAMPLE 26481//BPѵ��������
#define  TRAINING_TIME 1000

class BpNet
{
public:
	void train();//Bpѵ��
	double **p;  //���������
	double **t;  //����Ҫ�����
	double maxOut;
	double minOut;

	double *recognize(double *p);//Bpʶ��

	void writetrain(); //дѵ�����Ȩֵ
	void readtrain(); //��ѵ���õ�Ȩֵ����ʹ�Ĳ���ÿ��ȥѵ���ˣ�ֻҪ��ѵ����õ�Ȩֵ��������OK

	void readTrainCase();

	BpNet();
	virtual ~BpNet();

public:
	void init();
	double w[innode][hidenode];//�������Ȩֵ
	double w1[hidenode][outnode];//������Ȩֵ
	double b1[hidenode];//������㷧ֵ
	double b2[outnode];//�����㷧ֵ

	double rate_w; //Ȩֵѧϰ�ʣ������-������)
	double rate_w1;//Ȩֵѧϰ�� (������-�����)
	double rate_b1;//�����㷧ֵѧϰ��
	double rate_b2;//����㷧ֵѧϰ��

	double e;//������
	double error;//�����������
	double result[outnode];// Bp���
};

ifstream inTrainStream;
ifstream inTestStream;
ofstream outTestStream;

BpNet::BpNet()
{
	error = 1.0;
	e = 0.0;

	rate_w = 0.85;  //Ȩֵѧϰ�ʣ������--������)
	rate_w1 = 0.85; //Ȩֵѧϰ�� (������--�����)
	rate_b1 = 0.85; //�����㷧ֵѧϰ��
	rate_b2 = 0.85; //����㷧ֵѧϰ��
}

BpNet::~BpNet()
{
	for (int i = 0; i < TRAINSAMPLE; i++){
		delete[] p[i];
		delete[] t[i];
	}
	delete[] p;
	delete[] t;
}

void winit(double w[], int n) //Ȩֵ��ʼ��
{
	for (int i = 0; i < n; i++)
		w[i] = (2.0*(double)rand() / RAND_MAX) - 1;
}

void BpNet::init()
{
	winit((double*)w, innode*hidenode);
	winit((double*)w1, hidenode*outnode);
	winit(b1, hidenode);
	winit(b2, outnode);

	maxOut = INT_MIN;
	minOut = INT_MAX;

	p = new double*[TRAINSAMPLE];
	for (int i = 0; i < TRAINSAMPLE; i++){
		p[i] = new double[innode];
	}
	t = new double*[TRAINSAMPLE];
	for (int i = 0; i < TRAINSAMPLE; i++) {
		t[i] = new double[outnode];
	}
}

void BpNet::train()
{
	double pp[hidenode];//��������У�����
	double qq[outnode];//ϣ�����ֵ��ʵ�����ֵ��ƫ��
	double yd[outnode];//ϣ�����ֵ

	double x[innode]; //��������
	double x1[hidenode];//�������״ֵ̬
	double x2[outnode];//������״ֵ̬
	double o1[hidenode];//�����㼤��ֵ
	double o2[hidenode];//����㼤��ֵ

	for (int isamp = 0; isamp < TRAINSAMPLE; isamp++)//ѭ��ѵ��һ����Ʒ
	{
		for (int i = 0; i < innode; i++) {
			x[i] = p[isamp][i]; //���������
		}
		for (int i = 0; i < outnode; i++) {
			double mapValue = (t[isamp][i] - minOut) / (maxOut - minOut);
			yd[i] = mapValue; //�������������
		}

		//����ÿ����Ʒ������������׼
		for (int j = 0; j < hidenode; j++)
		{
			o1[j] = 0.0;
			for (int i = 0; i < innode; i++)
				o1[j] = o1[j] + w[i][j] * x[i];//���������Ԫ���뼤��ֵ
			x1[j] = 1.0 / (1 + exp(-o1[j] - b1[j]));//���������Ԫ�����
			//    if(o1[j]+b1[j]>0) x1[j]=1;
			//else x1[j]=0;
		}

		for (int k = 0; k < outnode; k++)
		{
			o2[k] = 0.0;
			for (int j = 0; j < hidenode; j++)
				o2[k] = o2[k] + w1[j][k] * x1[j]; //��������Ԫ���뼤��ֵ
			x2[k] = 1.0 / (1.0 + exp(-o2[k] - b2[k])); //��������Ԫ���
			//    if(o2[k]+b2[k]>0) x2[k]=1;
			//    else x2[k]=0;
		}

		for (int k = 0; k < outnode; k++)
		{
			qq[k] = (yd[k] - x2[k])*x2[k] * (1 - x2[k]); //ϣ�������ʵ�������ƫ��
			for (int j = 0; j < hidenode; j++)
				w1[j][k] += rate_w1*qq[k] * x1[j];  //��һ�ε�������������֮���������Ȩ
		}

		for (int j = 0; j < hidenode; j++)
		{
			pp[j] = 0.0;
			for (int k = 0; k < outnode; k++)
				pp[j] = pp[j] + qq[k] * w1[j][k];
			pp[j] = pp[j] * x1[j] * (1 - x1[j]); //�������У�����

			for (int i = 0; i < innode; i++)
				w[i][j] += rate_w*pp[j] * x[i]; //��һ�ε�������������֮���������Ȩ
		}

		for (int k = 0; k < outnode; k++)
		{
			e += fabs(yd[k] - x2[k])*fabs(yd[k] - x2[k]); //���������
		}
		error = e / 2.0;

		for (int k = 0; k < outnode; k++)
			b2[k] = b2[k] + rate_b2*qq[k]; //��һ�ε�������������֮�������ֵ
		for (int j = 0; j < hidenode; j++)
			b1[j] = b1[j] + rate_b1*pp[j]; //��һ�ε�������������֮�������ֵ
	}
}

double *BpNet::recognize(double *p)
{
	double x[innode]; //��������
	double x1[hidenode]; //�������״ֵ̬
	double x2[outnode]; //������״ֵ̬
	double o1[hidenode]; //�����㼤��ֵ
	double o2[hidenode]; //����㼤��ֵ

	for (int i = 0; i < innode; i++)
		x[i] = p[i];

	for (int j = 0; j < hidenode; j++)
	{
		o1[j] = 0.0;
		for (int i = 0; i < innode; i++)
			o1[j] = o1[j] + w[i][j] * x[i]; //���������Ԫ����ֵ
		x1[j] = 1.0 / (1.0 + exp(-o1[j] - b1[j])); //���������Ԫ���
		//if(o1[j]+b1[j]>0) x1[j]=1;
		//    else x1[j]=0;
	}

	for (int k = 0; k < outnode; k++)
	{
		o2[k] = 0.0;
		for (int j = 0; j < hidenode; j++)
			o2[k] = o2[k] + w1[j][k] * x1[j];//��������Ԫ����ֵ
		x2[k] = 1.0 / (1.0 + exp(-o2[k] - b2[k]));//��������Ԫ���
		//if(o2[k]+b2[k]>0) x2[k]=1;
		//else x2[k]=0;
	}

	for (int k = 0; k < outnode; k++)
	{
		result[k] = x2[k] * (maxOut - minOut) + minOut;
	}
	return result;
}

void BpNet::writetrain()
{
	FILE *stream0;
	FILE *stream1;
	FILE *stream2;
	FILE *stream3;
	int i, j;
	//�������Ȩֵд��
	if ((stream0 = fopen("w.txt", "w+")) == NULL)
	{
		cout << "�����ļ�ʧ��!";
		exit(1);
	}
	for (i = 0; i < innode; i++)
	{
		for (j = 0; j < hidenode; j++)
		{
			fprintf(stream0, "%f\n", w[i][j]);
		}
	}
	fclose(stream0);

	//������Ȩֵд��
	if ((stream1 = fopen("w1.txt", "w+")) == NULL)
	{
		cout << "�����ļ�ʧ��!";
		exit(1);
	}
	for (i = 0; i < hidenode; i++)
	{
		for (j = 0; j < outnode; j++)
		{
			fprintf(stream1, "%f\n", w1[i][j]);
		}
	}
	fclose(stream1);

	//������㷧ֵд��
	if ((stream2 = fopen("b1.txt", "w+")) == NULL)
	{
		cout << "�����ļ�ʧ��!";
		exit(1);
	}
	for (i = 0; i < hidenode; i++)
		fprintf(stream2, "%f\n", b1[i]);
	fclose(stream2);

	//�����㷧ֵд��
	if ((stream3 = fopen("b2.txt", "w+")) == NULL)
	{
		cout << "�����ļ�ʧ��!";
		exit(1);
	}
	for (i = 0; i < outnode; i++)
		fprintf(stream3, "%f\n", b2[i]);
	fclose(stream3);

}

void BpNet::readtrain()
{
	FILE *stream0;
	FILE *stream1;
	FILE *stream2;
	FILE *stream3;
	int i, j;

	//�������Ȩֵ����
	if ((stream0 = fopen("w.txt", "r")) == NULL)
	{
		cout << "���ļ�ʧ��!";
		exit(1);
	}
	float  wx[innode][hidenode];
	for (i = 0; i < innode; i++)
	{
		for (j = 0; j < hidenode; j++)
		{
			fscanf(stream0, "%f", &wx[i][j]);
			w[i][j] = wx[i][j];
		}
	}
	fclose(stream0);

	//������Ȩֵ����
	if ((stream1 = fopen("w1.txt", "r")) == NULL)
	{
		cout << "���ļ�ʧ��!";
		exit(1);
	}
	float  wx1[hidenode][outnode];
	for (i = 0; i < hidenode; i++)
	{
		for (j = 0; j < outnode; j++)
		{
			fscanf(stream1, "%f", &wx1[i][j]);
			w1[i][j] = wx1[i][j];
		}
	}
	fclose(stream1);

	//������㷧ֵ����
	if ((stream2 = fopen("b1.txt", "r")) == NULL)
	{
		cout << "���ļ�ʧ��!";
		exit(1);
	}
	float xb1[hidenode];
	for (i = 0; i < hidenode; i++)
	{
		fscanf(stream2, "%f", &xb1[i]);
		b1[i] = xb1[i];
	}
	fclose(stream2);

	//�����㷧ֵ����
	if ((stream3 = fopen("b2.txt", "r")) == NULL)
	{
		cout << "���ļ�ʧ��!";
		exit(1);
	}
	float xb2[outnode];
	for (i = 0; i < outnode; i++)
	{
		fscanf(stream3, "%f", &xb2[i]);
		b2[i] = xb2[i];
	}
	fclose(stream3);
}

void BpNet::readTrainCase(){
	int id;
	char comma;
	for (int caseNum = 0; caseNum < TRAINSAMPLE; caseNum++){
		inTrainStream >> id >> comma;
		for (int i = 0; i < innode; i++) {
			inTrainStream >> p[caseNum][i] >> comma;
		}
		for (int i = 0; i < outnode; i++) {
			inTrainStream >> t[caseNum][i];
			if (maxOut < t[caseNum][i])
				maxOut = t[caseNum][i];
			if (minOut > t[caseNum][i])
				minOut = t[caseNum][i];
		}
		while (inTrainStream.peek() == '\n') {
			inTrainStream.get();
		}
	}
}

void readTestCase(double(&testCase)[innode], int &id) {
	char comma;
	inTestStream >> id >> comma;
	for (int i = 0; i < innode - 1; i++) {
		inTestStream >> testCase[i] >> comma;
	}
	inTestStream >> testCase[innode - 1];
	while (inTestStream.peek() == '\n') {
		inTestStream.get();
	}

}

double X[TRAINSAMPLE][innode] = {
	{ 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 }, { 1, 1, 1 }
};
//�����������  
double Y[TRAINSAMPLE][outnode] = {
	{ 0 }, { 0.1429 }, { 0.2857 }, { 0.4286 }, { 0.5714 }, { 0.7143 }, { 0.8571 }, { 1.0000 }
};


int main()
{
	BpNet bp;
	bp.init();
	int times = 0;
	string str;
	inTrainStream.open("train_temp.csv");

	inTestStream.open("test_temp.csv");
	getline(inTestStream, str);

	outTestStream.open("submission.csv");
	outTestStream << "Id,reference\n" << "0,20.0000\n";

	bp.readTrainCase();
	while (bp.error > 0.0001)
	{
		if (times >= TRAINING_TIME)
			break;
		bp.e = 0.0;
		times++;
		bp.train();
		cout << "Times=" << times << " error=" << bp.error << endl;
	}
	cout << "trainning complete..." << endl;
	while (!inTestStream.eof()) {
		int testID;
		double testCase[innode];
		readTestCase(testCase, testID);
		bp.recognize(testCase);
		outTestStream << testID << "," << bp.result[0] << "\n";
	}

	cout << endl;
	return 0;
}