#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <string.h>
#include <time.h>
#include "para.h"
#include "Tracker.h"
#include "fft.h"

int main()
{
	int i,j,h,l,resize_image,scale,heightscale,widthscale,startframe,endframe,vert_delta, horiz_delta,window_sz[2],sz[2],**pos;
	double *cos_window,*han0,*han1,*fea,*xf,*response,*tmp1,*tmp2,output_sigma,*labels;
	complexdouble *complex_labels,*complex_xf,*complex_kf,*alphaf,*model_alphaf,*model_xf,*complex_fea,*complex_zf,*complex_tmp1,*complex_tmp2,*complex_kzf;
	float *imgreyfloat,*imgreyresizefloat;
	float t1,t2,t3,t4,max;
	int loc;
	uint8 * im,*rdata,*gdata,*bdata,*imgrey,*imgreyresize,*patch;
	FILE *fp;
	clock_t start,end,start1,end1;
	//color
	//char img_name[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\Trellis\\im%d.raw";
	//char img_name[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\fasthand5\\im%d.raw";
	//char img_name[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\hand2\\im%d.raw";
	char img_name[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\fasthand7\\im%d.raw";
	//char img_name[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\fasthand10\\im%d.raw";
	//gray
	//char img_name1[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\Trellis\\imresize%d.raw";
	//char img_name1[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\fasthand5\\imresize%d.raw";
	//char img_name1[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\hand2\\imresize%d.raw";
	//char img_name1[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\fasthand7\\imresize%d.raw";
	//char img_name1[1024]="E:\\project\\TrackingDemo\\TrackingDemo\\data\\fasthand10\\imresize%d.raw";
	char img_path[1024];
	
	//init tracking parameters
	tracker_para* trackerpara=(tracker_para*)malloc(sizeof(tracker_para));
	target_para* targetpara=(target_para*)malloc(sizeof(target_para));
	
	para_init(trackerpara);
	target_para_init(targetpara);
	
	im=(uint8 *)malloc((*targetpara).height*(*targetpara).width*(*targetpara).dim*sizeof(uint8));
	rdata=(uint8 *)malloc((*targetpara).height*(*targetpara).width*sizeof(uint8));
	gdata=(uint8 *)malloc((*targetpara).height*(*targetpara).width*sizeof(uint8));
	bdata=(uint8 *)malloc((*targetpara).height*(*targetpara).width*sizeof(uint8));
	imgrey=(uint8 *)malloc((*targetpara).height*(*targetpara).width*sizeof(uint8));
	imgreyfloat=(float *)malloc((*targetpara).height*(*targetpara).width*sizeof(float));

	//frame number
	startframe=(*targetpara).startframe;
	endframe=(*targetpara).endframe;

	pos=(int**)malloc((endframe-startframe+1)*sizeof(int*));
	for(i=0;i<(endframe-startframe+1);i++)
	{
		pos[i]=(int *)malloc(2*sizeof(int));
	}
	
	resize_image=0;
	scale=1;
	output_sigma=0;

	//read buffer
	sprintf(img_path,img_name,startframe);
	readdata(img_path,im,(*targetpara).height*(*targetpara).width*(*targetpara).dim);
	
	//writedata("Cim.raw",im,(*targetpara).height*(*targetpara).width*(*targetpara).dim);

	while(sqrt((double)(*targetpara).target_sz[0]*(*targetpara).target_sz[1])>= 100)
	{
		scale=scale*2;	
		(*targetpara).target_sz[0] = floor((double)(*targetpara).target_sz[0] / 2);
		(*targetpara).target_sz[1] = floor((double)(*targetpara).target_sz[1] / 2);
		(*targetpara).pos[0]=floor((double)(*targetpara).pos[0]/2);
		(*targetpara).pos[1]=floor((double)(*targetpara).pos[1]/2);
		resize_image=1;
	}
	
	//search window size
	window_sz[0] = floor((double)(*targetpara).target_sz[0] * (1 + (*trackerpara).padding));
	window_sz[1] = floor((double)(*targetpara).target_sz[1] * (1 + (*trackerpara).padding));

	//rgb2grey
	if((*targetpara).dim==3)
	{
		rgb2grey(im,im+(*targetpara).height*(*targetpara).width,im+2*(*targetpara).height*(*targetpara).width,(*targetpara).height,(*targetpara).width,imgrey);
		//writedata("Cimgrey.raw",imgrey,(*targetpara).height*(*targetpara).width);
	}
	
	//scale img
	heightscale = (int)((*targetpara).height/scale);
	widthscale = (int)((*targetpara).width/scale);

	imgreyresize=(uint8 *)malloc(heightscale*widthscale*sizeof(uint8));
	imgreyresizefloat=(float *)malloc(heightscale*widthscale*sizeof(float));
	//resize

	if(resize_image)
	{	
		for(i=0;i<(*targetpara).height*(*targetpara).width;i++)
		{
			imgreyfloat[i]=imgrey[i]*1.0f;
		}
		resize_im(imgreyfloat, (*targetpara).width, (*targetpara).height, 1, imgreyresizefloat, widthscale,heightscale);
		for(i=0;i<heightscale*widthscale;i++)
		{
			imgreyresize[i]=(uint8)imgreyresizefloat[i];
		}
		//writedata("Cimgreyresize.raw",imgreyresize,heightscale*widthscale);
	}
	else
	{	
		memcpy(imgreyresize,imgrey,heightscale*widthscale*sizeof(uint8));
		//for(i=0;i<heightscale*widthscale;i++)
		//	imgreyresize[i]=imgrey[i];
	}

	//sprintf(img_path,img_name1,startframe);
	//writedata(img_path,imgreyresize,heightscale*widthscale);

	//sprintf(img_path,img_name1,startframe);
	//readdata(img_path,imgreyresize,heightscale*widthscale);
	
	//readdata("Mimgreyresize.raw",imgreyresize,heightscale*widthscale);

	output_sigma = sqrt((float)(*targetpara).target_sz[0]*(*targetpara).target_sz[1])*(*trackerpara).output_sigma_factor/(*trackerpara).cell_size;

	//after compute HOG,window size resize 
	sz[0]=floor((double)window_sz[0]/ (*trackerpara).cell_size);
	sz[1]=floor((double)window_sz[1]/ (*trackerpara).cell_size);
	
	labels=(double*)malloc(sz[0]*sz[1]*sizeof(double));
	gaussian_shaped_labels(output_sigma,sz,labels);
	#if 1
	fp=fopen("labels.txt","w");
	for(i=0; i<sz[0]; i++)
	{
		for(j=0; j<sz[1]; j++)
		{
			fprintf(fp,"%f ",labels[i*sz[1]+j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	#endif
	complex_labels=(complexdouble*)malloc(sz[0]*sz[1]*sizeof(complexdouble));
	for(i=0;i<sz[0]*sz[1];i++)
	{
		complex_labels[i].real=labels[i];
		complex_labels[i].imag=0;
	}
	//store pre-computed cosine window
	
	//fft2d
	fft_2D(sz[0],sz[1],log((double)sz[0])/log((double)2),log((double)sz[1])/log((double)2),complex_labels,0);
	//for(i=0; i<sz[0]*sz[1]; i++)
	//{
	//		complex_labels[i].real=round(complex_labels[i].real*10e3)/10e3;
	//		complex_labels[i].imag=round(complex_labels[i].imag*10e3)/10e3;
	//}
	#if 1
	fp=fopen("complex_labels.txt","w");
	for(i=0; i<sz[0]; i++)
	{
		for(j=0; j<sz[1]; j++)
		{
			fprintf(fp,"%f\n%f\n",complex_labels[i*sz[1]+j].real,complex_labels[i*sz[1]+j].imag);
			//fprintf(fp,"%f ",complex_labels[i*sz[1]+j].real);
		}
		//fprintf(fp,"\n");
	}
	fclose(fp);
	#endif
	//store pre-computed cosine window
	han0=(double*)malloc(sz[0]*sizeof(double));
	hann(sz[0],han0);
	#if 0
	fp=fopen("han0.txt","w");
	for(i=0; i<sz[0]; i++)
	{
			fprintf(fp,"%f\n",han0[i]);
	}
	fclose(fp);
	#endif	
	//printf("\n%.40lf\n%.40lf\n%.40lf\n",han0[0],han0[1],han0[2]);
	han1=(double*)malloc(sz[1]*sizeof(double));
	hann(sz[1],han1);
	#if 0
	fp=fopen("han1.txt","w");
	for(i=0; i<sz[1]; i++)
	{
			fprintf(fp,"%f\n",han1[i]);
	}
	fclose(fp);
	#endif	
	
	cos_window=(double*)malloc(sz[0]*sz[1]*sizeof(double));
	for(i=0;i<sz[0];i++)
		for(j=0;j<sz[1];j++)
		{
			//cos_window[i*sz[1]+j]=round(han0[i]*han1[j]*10e3)/10e3;
			cos_window[i*sz[1]+j]=han0[i]*han1[j];
		}

	#if 0
	fp=fopen("cos_window.txt","w");
	for(i=0; i<sz[0]; i++)
	{
		for(j=0; j<sz[1]; j++)
		{
			fprintf(fp,"%f ",cos_window[i*sz[1]+j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	#endif

	//obtain a subwindow for training at newly estimated target position
	patch=(uint8*)malloc(window_sz[0]*window_sz[1]*sizeof(uint8));

	//readdata("E:\\KCF\\tracker_release2\\imptach.raw",imgreyresize,heightscale*widthscale);
	//printf("%d,%d,%d,",imgreyresize[0],imgreyresize[1],imgreyresize[2]);
	get_subwindow(imgreyresize, (*targetpara).pos,window_sz,heightscale,widthscale,patch);
	//writedata("Cpatch.raw",patch,window_sz[0]*window_sz[1]);
	
	
	xf=(double*)malloc(sz[0]*sz[1]*31*sizeof(double));
	memset(xf,0,sz[0]*sz[1]*31*sizeof(double));
	//readdata("Mpatch.raw",patch,window_sz[0]*window_sz[1]);
	get_features(patch,window_sz[0],window_sz[1],(*trackerpara).cell_size, cos_window,xf);
	#if 1
	sprintf(img_path,"xf%d.txt",startframe);
	fp=fopen(img_path,"w");
	for(i=0; i<31; i++)
	{
		for(j=0;j<sz[0];j++)
		{
			for(h=0;h<sz[1];h++)
			{
				fprintf(fp,"%f ",xf[i*sz[0]*sz[1]+j*sz[1]+h]);
			}
			fprintf(fp,"\n");
		}
		
	}
	fclose(fp);
	#endif
	complex_xf=(complexdouble*)malloc(sz[0]*sz[1]*31*sizeof(complexdouble));
	for(i=0;i<sz[0]*sz[1]*31;i++)
	{
		complex_xf[i].real=xf[i];
		complex_xf[i].imag=0;
	}
	for(i=0;i<31;i++)
	{
		fft_2D(sz[0],sz[1],log((double)sz[0])/log((double)2),log((double)sz[1])/log((double)2),complex_xf+i*sz[0]*sz[1],0);
	}
	//for(i=0; i<sz[0]*sz[1]*31; i++)
	//{
	//		complex_xf[i].real=round(complex_xf[i].real*10e3)/10e3;
	//		complex_xf[i].imag=round(complex_xf[i].imag*10e3)/10e3;
	//}
	#if 1
	sprintf(img_path,"complex_xf%d.txt",startframe);
	fp=fopen(img_path,"w");
	for(l=0;l<31;l++)
	{
		for(i=0; i<sz[0]; i++)
		{
			for(j=0;j<sz[1];j++)
			{
				//complex_xf[i*sz[1]+j+l*sz[0]*sz[1]].real=round(complex_xf[i*sz[1]+j+l*sz[0]*sz[1]].real*10e3)/10e3;
				//complex_xf[i*sz[1]+j+l*sz[0]*sz[1]].imag=round(complex_xf[i*sz[1]+j+l*sz[0]*sz[1]].imag*10e3)/10e3;
				fprintf(fp,"%f,%f\n",complex_xf[i*sz[1]+j+l*sz[0]*sz[1]].real,complex_xf[i*sz[1]+j+l*sz[0]*sz[1]].imag);
			}
		}
	}
	fclose(fp);
	#endif
	complex_kf=(complexdouble*)malloc(sz[0]*sz[1]*sizeof(complexdouble));

	linear_correlation(complex_xf,complex_xf, sz[0],sz[1],complex_kf);
	//for(i=0; i<sz[0]*sz[1]; i++)
	//{
	//		complex_kf[i].real=round(complex_kf[i].real*10e3)/10e3;
	//		complex_kf[i].imag=round(complex_kf[i].imag*10e3)/10e3;
	//}
	#if 1
	sprintf(img_path,"complex_kf%d.txt",startframe);
	fp=fopen(img_path,"w");
	for(i=0; i<sz[0]; i++)
	{
		for(j=0;j<sz[1];j++)
		{
				//complex_kf[i*sz[1]+j].real=round(complex_kf[i*sz[1]+j].real*10e3)/10e3;
				//complex_kf[i*sz[1]+j].imag=round(complex_kf[i*sz[1]+j].imag*10e3)/10e3;
				//fprintf(fp,"%f,%f ",complex_kf[i*sz[1]+j].real,complex_kf[i*sz[1]+j].imag);
				fprintf(fp,"%f\n",complex_kf[i*sz[1]+j].real);
				fprintf(fp,"%f\n",complex_kf[i*sz[1]+j].imag);
		}
	}
	fclose(fp);
	#endif
	alphaf=(complexdouble*)malloc(sz[0]*sz[1]*sizeof(complexdouble));
	model_alphaf=(complexdouble*)malloc(sz[0]*sz[1]*sizeof(complexdouble));
	model_xf=(complexdouble*)malloc(sz[0]*sz[1]*31*sizeof(complexdouble));
	for(i=0;i<sz[0]*sz[1];i++)
	{		
		complex_kf[i].real+= (*trackerpara).lambda;
		c_div(complex_labels[i],complex_kf[i],&alphaf[i]);
		//first frame, train with a single image
		model_alphaf[i] = alphaf[i];
		//model_alphaf[i].real=round(model_alphaf[i].real*10e3)/10e3;
		//model_alphaf[i].imag=round(model_alphaf[i].imag*10e3)/10e3;
	}
	#if 1
	sprintf(img_path,"model_alphaf%d.txt",startframe);
	fp=fopen(img_path,"w");
	for(i=0; i<sz[0]; i++)
	{
		for(j=0;j<sz[1];j++)
		{
				fprintf(fp,"%f\n%f\n",(float)model_alphaf[i*sz[1]+j].real,(float)model_alphaf[i*sz[1]+j].imag);
		}
	}
	fclose(fp);
	#endif
	for(i=0;i<sz[0]*sz[1]*31;i++)
	{	
		model_xf[i] = complex_xf[i];
		//model_xf[i].real=round(model_xf[i].real*10e3)/10e3;
		//model_xf[i].imag=round(model_xf[i].imag*10e3)/10e3;
	}

	#if 0
	sprintf(img_path,"model_xf%d.txt",startframe);
	fp=fopen(img_path,"w");
	for(i=0; i<sz[0]*sz[1]*31; i++)
	{
			fprintf(fp,"%f,%f\n",(float)model_xf[i].real,(float)model_xf[i].imag);
	}
	fclose(fp);
	#endif
	if(resize_image)
	{
		pos[0][0]=(*targetpara).pos[0]*scale;
		pos[0][1]=(*targetpara).pos[1]*scale;
	}else
	{
		pos[0][0]=(*targetpara).pos[0];
		pos[0][1]=(*targetpara).pos[1];	
	}
	fea=(double*)malloc(sz[0]*sz[1]*31*sizeof(double));
	complex_fea=(complexdouble*)malloc(sz[0]*sz[1]*sizeof(complexdouble));
	complex_tmp1=(complexdouble*)malloc(sz[0]*sz[1]*sizeof(complexdouble));
	complex_tmp2=(complexdouble*)malloc(sz[0]*sz[1]*sizeof(complexdouble));
	complex_zf=(complexdouble*)malloc(sz[0]*sz[1]*31*sizeof(complexdouble));
	complex_kzf=(complexdouble*)malloc(sz[0]*sz[1]*sizeof(complexdouble));
	start=clock();
	for(i=startframe+1;i<=endframe;i++)
	{	
		printf("%d frame\n",i);

		start1=clock();
		sprintf(img_path,img_name,i);
		readdata(img_path,im,(*targetpara).height*(*targetpara).width*(*targetpara).dim);
		end1=clock();
		printf("readdata=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		
		start1=clock();
		if((*targetpara).dim==3)
		{			
			rgb2grey(im,im+(*targetpara).height*(*targetpara).width,im+(*targetpara).height*(*targetpara).width*2,(*targetpara).height,(*targetpara).width,imgrey);
			//readdata(img_path,imgrey,(*targetpara).height*(*targetpara).width);
		}
		end1=clock();
		printf("rgb2gray=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);

		start1=clock();
		if(resize_image)
		{
				for(j=0;j<(*targetpara).height*(*targetpara).width;j++)
				{
					imgreyfloat[j]=imgrey[j]*1.0f;
				}
				resize_im(imgreyfloat, (*targetpara).width, (*targetpara).height, 1, imgreyresizefloat, widthscale,heightscale);
				for(j=0;j<heightscale*widthscale;j++)
				{
					imgreyresize[j]=(uint8)imgreyresizefloat[j];
				}
		}
		else
		{
			//for(j=0;j<heightscale*widthscale;j++)
			//	imgreyresize[j]=imgrey[j];
			memcpy(imgreyresize,imgrey,heightscale*widthscale*sizeof(uint8));
		}

		//sprintf(img_path,img_name1,i);
		//writedata(img_path,imgreyresize,heightscale*widthscale);

		//sprintf(img_path,img_name1,i);
		//readdata(img_path,imgreyresize,heightscale*widthscale);
		end1=clock();
		printf("imresize=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);

		start1=clock();
		get_subwindow(imgreyresize, (*targetpara).pos, window_sz,heightscale,widthscale,patch);
		end1=clock();
		printf("get_subwindow=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		memset(fea,0,sz[0]*sz[1]*31*sizeof(double));
		get_features(patch,window_sz[0],window_sz[1],(*trackerpara).cell_size, cos_window,fea);
		#if 0
		sprintf(img_path,"fea_predict%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]; j++)
		{
				for(h=0;h<sz[1];h++)
					fprintf(fp,"%f ",fea[j*sz[1]+h]);
				fprintf(fp,"\n");
		}
		fclose(fp);
		#endif
		end1=clock();
		printf("get_features=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		for(j=0;j<sz[0]*sz[1]*31;j++)
		{
			complex_zf[j].real=fea[j];
			complex_zf[j].imag=0;
		}
		end1=clock();
		printf("copydata=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		for(j=0;j<31;j++)
		{
			fft_2D(sz[0],sz[1],log((double)sz[0])/log(2.0),log((double)sz[1])/log(2.0),complex_zf+j*sz[0]*sz[1],0);
		}
		//for(j=0;j<sz[0]*sz[1]*31;j++)
		//{
		//	complex_zf[j].real=round(complex_zf[j].real*10e3)/10e3;
		//	complex_zf[j].imag=round(complex_zf[j].imag*10e3)/10e3;
		//}
		#if 0
		sprintf(img_path,"complex_zf_predict%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]; j++)
		{
				for(h=0;h<sz[1];h++)
					fprintf(fp,"%f,%f ",complex_zf[j*sz[1]+h].real,complex_zf[j*sz[1]+h].imag);
				fprintf(fp,"\n");
		}
		fclose(fp);
		#endif
		end1=clock();
		printf("fft2d=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();

		linear_correlation(complex_zf, model_xf,sz[0],sz[1],complex_kzf);
		//for(j=0; j<sz[0]*sz[1]; j++)
		//{
		//	complex_kzf[j].real=round(complex_kzf[j].real*10e3)/10e3;
		//	complex_kzf[j].imag=round(complex_kzf[j].imag*10e3)/10e3;
		//}
		end1=clock();
		printf("correlation=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();		
		#if 0
		sprintf(img_path,"model_alphaf_predict%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]; j++)
			{
					for(h=0;h<sz[1];h++)
						fprintf(fp,"%f,%f ",model_alphaf[j*sz[1]+h].real,model_alphaf[j*sz[1]+h].imag);
					fprintf(fp,"\n");
			}
		fclose(fp);
		sprintf(img_path,"complex_kzf_predict%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]; j++)
			{
					for(h=0;h<sz[1];h++)
						fprintf(fp,"%f,%f ",complex_kzf[j*sz[1]+h].real,complex_kzf[j*sz[1]+h].imag);
					fprintf(fp,"\n");
			}
		fclose(fp);
		#endif
		for(j=0;j<sz[0]*sz[1];j++)
		{
			c_mul(model_alphaf[j],complex_kzf[j],&complex_tmp2[j]);
			//complex_tmp2[j].real=round(complex_tmp2[j].real*10e3)/10e3;
			//complex_tmp2[j].imag=round(complex_tmp2[j].imag*10e3)/10e3;
		}
		#if 0
		sprintf(img_path,"complex_tmp2_predict%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]; j++)
			{
					for(h=0;h<sz[1];h++)
					{
						fprintf(fp,"%f\n",complex_tmp2[j*sz[1]+h].real);
						fprintf(fp,"%f\n",complex_tmp2[j*sz[1]+h].imag);
					}
			}
		fclose(fp);
		#endif
		fft_2D(sz[0],sz[1],log((double)sz[0])/log(2.0),log((double)sz[1])/log(2.0),complex_tmp2,1);
		//for(j=0; j<sz[0]*sz[1]; j++)
		//{
		//	complex_tmp2[j].real=round(complex_tmp2[j].real*10e3)/10e3;
		//	complex_tmp2[j].imag=round(complex_tmp2[j].imag*10e3)/10e3;
		//}
		end1=clock();
		printf("ifft_2d=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		response=(double*)malloc(sz[0]*sz[1]*sizeof(double));
		computereal(complex_tmp2,response,sz[0]*sz[1]);  //equation for fast detection
		#if 0
		sprintf(img_path,"response%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]; j++)
		{
				for(h=0;h<sz[1];h++)
					fprintf(fp,"%f ",response[j*sz[1]+h]);
				fprintf(fp,"\n");
		}
		fclose(fp);
		#endif
		//target location is at the maximum response. we must take into
		//account the fact that, if the target doesn't move, the peak
		//will appear at the top-left corner, not at the center (this is
		//discussed in the paper). the responses wrap around cyclically.
		end1=clock();
		printf("computereal=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();

		maxfind(response,sz[0],sz[1],&vert_delta,&horiz_delta);
        if(vert_delta>sz[0]/2)  //wrap around to negative half-space of vertical axis
        {    
			vert_delta = vert_delta - sz[0];
		}
        if(horiz_delta>sz[1]/2)  //same for horizontal axis
        { 
			horiz_delta = horiz_delta -sz[1];
		}
		end1=clock();
		printf("maxfind=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		(*targetpara).pos[0]=(*targetpara).pos[0]+(*trackerpara).cell_size*(vert_delta); 
		(*targetpara).pos[1]=(*targetpara).pos[1]+(*trackerpara).cell_size*(horiz_delta);
		//printf("%dframe pos=%d,%d\n",i,(*targetpara).pos[0],(*targetpara).pos[1]);
		if(resize_image)
		{
			pos[i-(startframe+1)+1][0]=(*targetpara).pos[0]*scale;
			pos[i-(startframe+1)+1][1]=(*targetpara).pos[1]*scale;
		}else
		{
			pos[i-(startframe+1)+1][0]=(*targetpara).pos[0];
			pos[i-(startframe+1)+1][1]=(*targetpara).pos[1];		
		}
		//obtain a subwindow for training at newly estimated target position
		end1=clock();
		printf("computepos=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		#if 0
		sprintf(img_path,"im%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<heightscale; j++)
		{
			for(h=0;h<widthscale;h++)
			{
					fprintf(fp,"%d ",imgreyresize[j*widthscale+h]);
			}
			fprintf(fp,"\n");
		}
		fclose(fp);
		#endif
		get_subwindow(imgreyresize, (*targetpara).pos, window_sz,heightscale,widthscale,patch);
		#if 0
		sprintf(img_path,"patch%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<window_sz[0]; j++)
		{
			for(h=0;h<window_sz[1];h++)
			{
					fprintf(fp,"%d ",patch[j*window_sz[1]+h]);
			}
			fprintf(fp,"\n");
		}
		fclose(fp);
		#endif
		end1=clock();
		printf("get_subwindow=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		memset(fea,0,sz[0]*sz[1]*31*sizeof(double));
		get_features(patch,window_sz[0],window_sz[1],(*trackerpara).cell_size, cos_window,fea);
		#if 0
		sprintf(img_path,"fea%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]*sz[1]*31; j++)
			{
					fprintf(fp,"%f\n",fea[j]);
			}
		fclose(fp);
		#endif



		end1=clock();
		printf("get_features=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		for(j=0;j<sz[0]*sz[1]*31;j++)
		{
			complex_xf[j].real=fea[j];
			complex_xf[j].imag=0;
		}
		end1=clock();
		printf("copydata=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		for(j=0;j<31;j++)
		{
			fft_2D(sz[0],sz[1],log((double)sz[0])/log(2.0),log((double)sz[1])/log(2.0),complex_xf+j*sz[0]*sz[1],0);
		}
		//for(j=0;j<sz[0]*sz[1]*31;j++)
		//{
		//	complex_xf[j].real=round(complex_xf[j].real*10e3)/10e3;
		//	complex_xf[j].imag=round(complex_xf[j].imag*10e3)/10e3;
		//}
		#if 0
		sprintf(img_path,"complex_xf%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]*sz[1]*31; j++)
			{
					fprintf(fp,"%f,%f\n",(float)complex_xf[j].real,(float)complex_xf[j].imag);
			}
		fclose(fp);
		#endif
		end1=clock();
		printf("fft_2D=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		//Kernel Ridge Regression, calculate alphas (in Fourier domain)
		linear_correlation(complex_xf, complex_xf, sz[0],sz[1],complex_kf);
		//
		//for(j=0;j<sz[0]*sz[1];j++)
		//{
		//	complex_kf[j].real=round(complex_kf[j].real*10e3)/10e3;
		//	complex_kf[j].imag=round(complex_kf[j].imag*10e3)/10e3;
		//}
		#if 0
		fp=fopen("complex_kf.txt","w");
		for(j=0; j<sz[0]; j++)
			{
					for(h=0;h<sz[1];h++)
					{
						fprintf(fp,"%f,",complex_kf[j*sz[1]+h].real);
						fprintf(fp,"%f\n",complex_kf[j*sz[1]+h].imag);
					}
			}
		fclose(fp);
		#endif
		end1=clock();
		printf("correlation=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		for(j=0;j<sz[0]*sz[1];j++)
		{
			complex_kf[j].real+= (*trackerpara).lambda;
			c_div(complex_labels[j],complex_kf[j],&alphaf[j]); //equation for fast training
			//subsequent frames, interpolate model
			//model_alphaf[j].real=(1-(*trackerpara).interp_factor)*model_alphaf[j].real;
			//model_alphaf[j].imag=(1-(*trackerpara).interp_factor)*model_alphaf[j].imag;
			//alphaf[j].real=(*trackerpara).interp_factor*alphaf[j].real;
			//alphaf[j].imag=(*trackerpara).interp_factor*alphaf[j].imag;
			//c_plus(model_alphaf[j],alphaf[j],&model_alphaf[j]);
			model_alphaf[j].real=(1-(*trackerpara).interp_factor)*model_alphaf[j].real+(*trackerpara).interp_factor*alphaf[j].real;
			model_alphaf[j].imag=(1-(*trackerpara).interp_factor)*model_alphaf[j].imag+(*trackerpara).interp_factor*alphaf[j].imag;
			//model_alphaf[j].real=round(model_alphaf[j].real*10e3)/10e3;
			//model_alphaf[j].imag=round(model_alphaf[j].imag*10e3)/10e3;
		}
		#if 0
		fp=fopen("alphaf.txt","w");
		for(j=0; j<sz[0]; j++)
			{
					for(h=0;h<sz[1];h++)
					{
						fprintf(fp,"%f,",alphaf[j*sz[1]+h].real);
						fprintf(fp,"%f\n",alphaf[j*sz[1]+h].imag);
					}
			}
		fclose(fp);
		#endif
		#if 1
		sprintf(img_path,"model_alphaf%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]; j++)
			{
					for(h=0;h<sz[1];h++)
					{
						fprintf(fp,"%f\n",model_alphaf[j*sz[1]+h].real);
						fprintf(fp,"%f\n",model_alphaf[j*sz[1]+h].imag);
					}
			}
		fclose(fp);
		#endif
		end1=clock();
		printf("updata_alphaf_model=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		start1=clock();
		for(j=0;j<sz[0]*sz[1]*31;j++)
		{
			model_xf[j].real=(1-(*trackerpara).interp_factor)*model_xf[j].real;
			model_xf[j].imag=(1-(*trackerpara).interp_factor)*model_xf[j].imag;
			complex_xf[j].real=(*trackerpara).interp_factor*complex_xf[j].real;
			complex_xf[j].imag=(*trackerpara).interp_factor*complex_xf[j].imag;
			c_plus(model_xf[j],complex_xf[j],&model_xf[j]);
			//model_xf[j].real=round(model_xf[j].real*10e3)/10e3;
			//model_xf[j].imag=round(model_xf[j].imag*10e3)/10e3;
		}
		#if 0
		sprintf(img_path,"model_xf%d.txt",i);
		fp=fopen(img_path,"w");
		for(j=0; j<sz[0]*sz[1]*31; j++)
			{
					fprintf(fp,"%f,%f\n",(float)model_xf[j].real,(float)model_xf[j].imag);
			}
		fclose(fp);
		#endif
		end1=clock();
		printf("updata_xf_model=%lf ms\n",(double)(end1-start1)/CLOCKS_PER_SEC*1000);
		printf("\n");
	}
	end=clock();
	printf("total=%lf ms\n",(double)(end-start)/CLOCKS_PER_SEC*1000);
	for(i=0;i<(endframe-startframe+1);i++)
	{
		printf("%d frame pos=%d,%d\n",i+1,pos[i][0],pos[i][1]);
	}
	
	free(trackerpara);
	free(targetpara);
	free(im);
	free(rdata);
	free(gdata);
	free(bdata);
	free(imgrey);
	free(imgreyresize);
	free(patch);
	for(i=0;i<(endframe-startframe+1);i++)
	{
		free(pos[i]);
	}
	free(pos);
	free(labels);
	free(cos_window);
	free(han0);
	free(han1);
	free(fea);
	free(xf);
	free(response);

	free(complex_labels);
	free(complex_xf);
	free(complex_kf);
	free(alphaf);
	free(model_alphaf);
	free(model_xf);
	free(complex_fea);
	free(complex_zf);
	free(complex_tmp1);
	free(complex_tmp2);
	free(complex_kzf);
	
	system("pause");
	return 0;
}
