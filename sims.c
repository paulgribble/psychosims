// gcc -Wall -O3 -o sims sims.c nmsimplex.c -lm -fopenmp

#include "nmsimplex.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

/* DATA STRUCTURE AND UTILITY FUNCTIONS TO STORE DATA LOADED FROM FILE  */
//
// a struct to hold the data
 typedef struct {
   double **data; // data[0] = lateral hand position
                  // data[1] = binary responses {0,1} = {left,right}
   int data_n;    // number of trials
} datastruct;

// utility function to initialize our datastruct
datastruct *datastruct_allocate(int n) {
  datastruct *thedata = malloc(sizeof(datastruct));
  thedata->data_n = n;
  thedata->data = malloc(sizeof(double)*2);
  thedata->data[0] = malloc(sizeof(double)*n);
  thedata->data[1] = malloc(sizeof(double)*n);
  return thedata;
}

// utility function to free our datastruct
void datastruct_free(datastruct *thedata) {
  free(thedata->data[0]);
  free(thedata->data[1]);
  free(thedata->data);
  free(thedata);
}
/* ========================================================================= */

/* ================ THE LOGIT LINK FUNCTION AND ITS INVERSE ================ */
//
// logit link function
double logit(double y)
{
  return 1 / (1 + exp(-y));
}

// inverse logit function
double i_logit(double p, double b[])
{
  return (log(-p / (p - 1)) - b[0]) / b[1];
}
/* ========================================================================= */

/* OBJECTIVE FUNCTION : CALCULATE LIKELIHOOD OF DATA GIVEN A CANDIDATE MODEL */
//
// negative log-likelihood of the data given
// the model (parameter vector x[]) and extra stuff (void *extra)
double nll(double x[], void *extra)
{
  int i;
  double neg_log_lik = 0.0;
  double pos, p, y;
  int r;
  double **data = ((datastruct *)extra)->data;
  double data_n = ((datastruct *)extra)->data_n;
  for (i=0; i<data_n; i++) {
    pos = data[0][i];
    r = (int) data[1][i];
    y = x[0] + x[1]*pos;
    p = logit(y); // logit link function
    if (p>=1.0) {p = 1.0 - 1e-10;} // avoid numerical nasties
    if (p<=0.0) {p = 1e-10;}
    if (r==1) { neg_log_lik -= log(p); }
    else { neg_log_lik -= log(1-p); }
  }
  return neg_log_lik;
}
/* ========================================================================= */

typedef struct {
  int nreps;
  double b0;
  double b1;
  double bb0;
  double bb1;
} wrapperstruct;

/* WRAPPER FUNCTION TO BE THREADED */
//
void *wrapper(void *voidin) {
  wrapperstruct *in = voidin;
  double x[7] = {-20.0, -10.0, -5.0, 0.0, 5.0,  10.0, 20.0};
  double y, p, r, bmin;
  int nreps = in->nreps;
  double b[2];
  b[0] = in->b0;
  b[1] = in->b1;
  double *bb = malloc(2 * sizeof(double));
  bb[0] = in->bb0;
  bb[1] = in->bb1;
  datastruct *ds = datastruct_allocate(nreps*7); // simulated data
  // fill data with x values and 0s for responses
  int i, j, k, kk;
  k = 0;
  for (i=0; i<7; i++) {
    for (j=0; j<nreps; j++) {
      ds->data[0][k] = x[i];
      ds->data[1][k] = 0.0;
      k++;
    }
  }  
  // simulate responses nreps times at each x position
  kk = 0;
  for (j=0; j<7; j++) {
    for (k=0; k<nreps; k++) {
      y = b[0] + (b[1] * x[j]);
      p = logit(y);
      r = (double) rand() / RAND_MAX;
      if (r <= p) { ds->data[1][kk] = 1.0; }
      else { ds->data[1][kk] = 0.0; }
      kk++;
    }
  }
  bmin = simplex(nll, bb, 2, 1.0e-8, 1, NULL, ds);
  datastruct_free(ds);
  in->bb0 = bb[0];
  in->bb1 = bb[1];
  return NULL;
}
/* ========================================================================= */


/* ============================ THE MAIN FUNCTION ========================== */
//
int main(int argc, char *argv[]) {
  
  int nsims = 10;
  int nreps = 8;
  double b0 = 0.0;
  double b1 = 0.40;
  int i, j, k;

  if (argc > 1) {
    nsims = atoi(argv[1]);
    nreps = atoi(argv[2]);
    b0 = atof(argv[3]);
    b1 = atof(argv[4]);
  }
  
  // y = b0 + (b1 * x)
  // p = 1 / (1 + exp(-y))
     
  srand((unsigned)time(NULL));
  // blow first value
  rand();

  wrapperstruct w[nsims];

#pragma omp parallel for
  for (i=0; i<nsims; i++) {
    w[i].nreps = nreps;
    w[i].b0 = b0;            // actual psychometric curve
    w[i].b1 = b1;            // actual psychometric curve
    w[i].bb0 = 0.0;          // simulated esimate
    w[i].bb1 = 0.40;         // simulated esimate
    wrapper(&w[i]);
  }

  double bb[2];
  double x25, x75, acuity;
  for (i=0; i<nsims; i++) {
    bb[0] = w[i].bb0;
    bb[1] = w[i].bb1;
    x25 = i_logit(0.25, bb);
    x75 = i_logit(0.75, bb);
    acuity = x75-x25;
    printf("%8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f\n",
	   bb[0], bb[1], -bb[0]/bb[1], bb[1]/4, x25, x75, acuity);
  }

  return 0;
}
