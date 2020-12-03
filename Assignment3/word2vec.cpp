#include <bits/stdc++.h>
using namespace std;
typedef long double ld;

int vocab_size;
int dimensions;
ld learning_rate;
int iterations;
int num_pairs;
vector<pair<int,int>> word_pairs;		// stores the training pairs
vector<vector<ld>> input_matrix;
vector<vector<ld>> output_matrix;
vector<ld> softmax_vals;				// stores the softmax values for output neurons

/* returns the softmax values for the input vector of values */
void getSoftmaxValues()
{
	int size = softmax_vals.size();
	ld total = 0;
	for(int i=0;i<size;i++)
	{
		ld val = softmax_vals[i];
		ld temp = exp(val);
		total += temp;
	}
	for(int i=0;i<size;i++)
	{
		ld val = softmax_vals[i];
		ld temp = exp(val);
		softmax_vals[i] = temp/total;
	}
}

void train()
{
	for(int i=0;i<iterations;i++)
	{
		for(int j=0;j<num_pairs;j++)
		{
			int input_word = word_pairs[j].first;			// index of input word
			int output_word = word_pairs[j].second;			// index of output word
			int neg=0;										// stores number of negative updates

			/* 	hidden layer values are equal to the row values of the input matrix
				corresponding to the index of input word */
			ld hidden[dimensions];							
			for(int k=0;k<dimensions;k++)
				hidden[k] = input_matrix[input_word-1][k];

			/* calculates the output of neurons in this vector softmax_vals */
			softmax_vals.resize(vocab_size);
			for(int k=0;k<vocab_size;k++)
			{
				ld temp = 0;
				for(int l=0;l<dimensions;l++)
					temp += hidden[l]*output_matrix[l][k];
				softmax_vals[k] = temp;
			}
			getSoftmaxValues();									// converts output to probabilities

			/* stores errors corresponding to each output neuron (y_k - t_k) */
			ld errors[vocab_size];								
			for(int k=0;k<vocab_size;k++)
			{
				if(k==output_word-1)
					errors[k] = softmax_vals[k]-1;
				else
					errors[k] = softmax_vals[k];
			}

			/* weight update rule on input matrix(input-hidden) */
			for(int k=0;k<dimensions;k++)
			{
				ld EH = 0;
				for(int l=0;l<vocab_size;l++)
					EH += errors[l]*output_matrix[k][l];
				ld delta = learning_rate*EH;
				input_matrix[input_word-1][k] -= delta;
				if(delta>0)
					neg++;
			}

			/* weight update rule on output matrix(hidden-output) */
			for(int k=0;k<dimensions;k++)
			{
				ld delta = learning_rate*errors[output_word-1]*hidden[k];
				output_matrix[k][output_word-1] -= delta;
				if(delta>0)
					neg++;
			}

			/* required printing, total no of weight updates are 2*(dimensions of hidden layer) */
			cout<<i+1<<" "<<j+1<<" "<<neg<<" "<<2*dimensions-neg<<endl;
		}
	}
}

int main()
{
	cin>>vocab_size;
	cin>>dimensions;
	cin>>learning_rate;
	cin>>iterations;
	cin>>num_pairs;

	for(int i=0;i<num_pairs;i++)
	{
		int id,a,b;cin>>id>>a>>b;
		word_pairs.push_back({a,b});
	}

	input_matrix.resize(vocab_size, vector<ld>(dimensions, 0.5));
	output_matrix.resize(dimensions, vector<ld>(vocab_size, 0.5));

	train();
	return 0;
}