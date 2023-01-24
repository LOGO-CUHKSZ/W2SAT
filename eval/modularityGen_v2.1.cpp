/*
Modularity-based k-CNF Generator.
Version 2.1
Authors:
  - Jesús Giráldez-Cru (IIIA-CSIC)
  - Jordi Levy (IIIA-CSIC)

Contact: jgiraldez@iiia.csic.es

	Copyright (C) 2015 J. Giráldez-Cru, J. Levy

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>

using namespace std;

int k = 3; // k-CNF
int seed = 0;
int n = 10000;	// number of variables
int m = 42500;	// number of clauses
double Q = 0.8; // modularity
int c = 80;		// number of communities

double P;

char *output = NULL;

void printUsage(char *app)
{
	cerr << "  Usage: " << app << " [options]" << endl;
	cerr << "    -n <int>   :  number of variables (10000)" << endl;
	cerr << "    -m <int>   :  number of clauses (42500)" << endl;
	cerr << "    -c <int>   :  number of communities (80)" << endl;
	cerr << "    -Q <float> :  modularity: Q (0.8)" << endl;
	cerr << "    -k <int>   :  number of literals by clause: k-CNF (3)" << endl;
	cerr << "    -s <int>   :  seed (0)" << endl;
	cerr << "    -o <string>:  output file (stdout)" << endl;
	cerr << "  Restrictions:" << endl;
	cerr << "    1. c must be greater than 1" << endl;
	cerr << "    2. Q must be in the interval (0,1)" << endl;
	cerr << "	 3. k must be greater than 1" << endl;
	cerr << "    4. c must be greater or equal than k" << endl;
	cerr << "    5. (c*k) must be smaller or equal than n" << endl;
	exit(-1);
}

void parseArgs(int argc, char **argv)
{
	int opt;
	while ((opt = getopt(argc, argv, "n:m:c:Q:s:k:?ho:")) != -1)
	{
		switch (opt)
		{
		case 'n':
			n = atoi(optarg);
			break;
		case 'm':
			m = atoi(optarg);
			break;
		case 'c':
			c = atoi(optarg);
			if (c <= 1 /*|| c >= xxx*/)
			{
				cerr << "WARNING: c must be greater than 1" << endl;
				cerr << "  c changed to 80 (default value)" << endl;
				c = 80;
			}
			break;
		case 'Q':
			Q = atof(optarg);
			if (Q <= 0 || Q >= 1)
			{
				cerr << "WARNING: Q must be in the interval (0,1)" << endl;
				cerr << "  Q changed to 0.8 (default value)" << endl;
				Q = 0.8;
			}
			break;
		case 's':
			seed = atoi(optarg);
			break;
		case 'k':
			k = atoi(optarg);
			if (k < 2)
			{
				cerr << "WARNING: k must be greater than 1" << endl;
				cerr << "  k changed to 3 (default value)" << endl;
				k = 3;
			}
			break;
		case 'o':
			output = optarg;
			break;
		case 'h':
		case '?':
			printUsage(argv[0]);
			break;
		default:
			cerr << "ERROR: Incorrect argument: " << optarg << endl;
			printUsage(argv[0]);
		}
	}

	if (c < k)
	{
		cerr << "ERROR: c (c=" << c << ") must be greater or equal than k (k=" << k << ")" << endl;
		cerr << "  Execution failed." << endl;
		exit(-1);
	}

	if (c * k > n)
	{
		cerr << "ERROR: c*k (c=" << c << ", k=" << k << ") must be less or equal than n (n=" << n << ")" << endl;
		cerr << "  Execution failed." << endl;
		exit(-1);
	}
}

void computeProbability()
{
	P = Q + 1 / (double)c;
}

void computeN2C(vector<int> &n2c)
{
	int rn;
	double rd;

	rn = rand();
	rd = ((double)rn) / (RAND_MAX);
	if (rd <= P)
	{ // All variables in the same community
		rn = rand();
		for (int i = 0; i < k; i++)
			n2c[i] = rn % c;
	}
	else
	{ // All variables in distict communitites
		for (int i = 0; i < k; i++)
		{
			bool used = false;
			do
			{
				used = false;
				rn = rand();
				for (int j = 0; j < i && !used; j++)
				{
					if (n2c[j] == rn % c)
						used = true;
				}
			} while (used);
			n2c[i] = rn % c;
		}
	}
}

void computeClause(vector<int> &n2c, vector<int> &clause)
{
	int rn;
	for (int j = 0; j < k; j++)
	{
		// Random variable in the community
		//   avoiding tautologies with previous literals
		int var;
		bool tautology = false;
		do
		{
			tautology = false;
			rn = rand();
			var = rn % (n2c[j] * n / c - (n2c[j] + 1) * n / c) + n2c[j] * n / c + 1;
			for (int l = 0; l < j && !tautology; l++)
			{
				if (abs(clause[l]) == var)
				{
					tautology = true;
				}
			}
		} while (tautology);

		// Polarity of the variable
		if (rn > (RAND_MAX / 2))
			var = -var;

		clause[j] = var;
	}
}

int main(int argc, char **argv)
{

	FILE *fout;

	// Parse arguments
	parseArgs(argc, argv);

	if (output != NULL)
	{
		fout = fopen(output, "w");
	}

	// Compute the probability P, according to k, c and Q
	computeProbability();

	// Print header
	if (output != NULL)
	{
		fprintf(fout, "c Modularity-based k-CNF Generator (V2.0). J. Giraldez-Cru and J. Levy\n");
		fprintf(fout, "c   value n = %d\n", n);
		fprintf(fout, "c   value m = %d\n", m);
		fprintf(fout, "c   value k = %d\n", k);
		fprintf(fout, "c   value Q = %f\n", Q);
		fprintf(fout, "c   value c = %d\n", c);
		fprintf(fout, "c   value seed = %d\n", seed);
		fprintf(fout, "p cnf %d %d\n", n, m);
	}
	else
	{
		cout << "c Modularity-based k-CNF Generator (V2.0). J. Giraldez-Cru and J. Levy" << endl;
		cout << "c   value n = " << n << endl;
		cout << "c   value m = " << m << endl;
		cout << "c   value k = " << k << endl;
		cout << "c   value Q = " << Q << endl;
		cout << "c   value c = " << c << endl;
		cout << "c   value seed = " << seed << endl;
		cout << "p cnf " << n << " " << m << endl;
	}

	int rn;	   // random number
	double rd; // random double between 0 and 1
	srand(seed);

	// Iterate for each clause
	for (int i = 0; i < m; i++)
	{

		// n2c is the community of each literal
		vector<int> n2c(k, 0);
		computeN2C(n2c);

		// Compute the clause
		vector<int> clause(k);
		computeClause(n2c, clause);

		// Print the clause
		for (int j = 0; j < k; j++)
		{
			if (output != NULL)
			{
				fprintf(fout, "%d ", clause[j]);
			}
			else
			{
				cout << clause[j] << " ";
			}
		}

		if (output != NULL)
		{
			fprintf(fout, "0\n");
		}
		else
		{
			cout << "0" << endl;
		}
	}

	if (output != NULL)
	{
		fclose(fout);
	}
}
