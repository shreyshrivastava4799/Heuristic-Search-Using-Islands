#include <queue>
#include <iostream>
#include <bits/stdc++.h>
#include <unistd.h>
#include <ctime>
#include <limits.h>

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

typedef struct _State
{
	int x;
	int y;
	double g;
	double h;
	struct _State *prnt;
	bool islandParent = false;


} State;

static float weight = 1;
int expanded_states = 0;

string destinationPath;
int experimentNumber;


// The operator treats its field as max priority queue so the one thats is greater 
// gets on the starting of queue, now we want those state that have lesser f values 
// in the front, so 'b' should be greater than 'a' in this operator check if 'b' has lesser
// f value . Now this "less than operator" ideally gives true when 'a' is less than 'b'
// but as we want them the one with less f value to be treated as of more priority we have 
// inverted the sign of equality

struct stateComparator{
	bool operator()(State* &a, State* &b)
	{
		// In general is to check for some attribute of State
		// return a.null_attribute < b.null_attribute;
		return a->g + weight*a->h > b->g + weight*b->h;
	}
};


class PathPlanner
{
		
 
public:

	PathPlanner(State start, State end, Mat img, vector<State> islands)
	{
		this->end = end;
		this->img = img;
		imgRows = img.rows;
		imgCols = img.cols;
		this->start = start;
        this->islands = islands;

	}

	~PathPlanner()
	{
		for (int i = 0; i < imgRows; ++i)
			for (int j = 0; j < imgCols; ++j)
			{
				if( record[i][j]!=NULL && !(i==start.x && j==start.y) && !(i==end.x && j==end.y) )
					delete record[i][j];	
			}
		// cout<<"Delete Successful"<<endl;
	}

	double getPath()
	{
		bool reached = false;

		Mat closed = img.clone();
		Mat visited = img.clone();
		Mat final = img.clone();
		// Show start and end points
		circle(final, Point(start.x, start.y), 5, Vec3b(0,0,255), 1);
		circle(final, Point(end.x, end.y), 5, Vec3b(0,255,255), 1);
		// imshow("Planning Problem", final);
		// waitKey(0);

		// Allocation of record for states
		record = new State**[imgRows];
		for (int i = 0; i < imgRows; ++i)
			record[i] = new State*[imgCols];
		
		for (int i = 0; i < imgRows; ++i)
			for (int j = 0; j < imgCols; ++j)
				record[i][j] = NULL;

		// Priority queue of pointers as open list
		priority_queue< State*, vector<State*>, stateComparator > pq;

		// Pushing start state into open list
		start.g = 0;
		start.h = getHeuristic(start);
		start.prnt = NULL;
		pq.push(&start);
		record[start.x][start.y] = &start;
		visited.at<Vec3b>(start.y,start.x) = Vec3b(0,0,255);
		
		int count = 0;
		while(!pq.empty() && !reached )
		{
			
			State *front = pq.top();
			pq.pop();

			expanded_states++;
			closed.at<Vec3b>(front->y,front->x) = Vec3b(0,255,0);
			
			// Visualisation of visited and closed states 
			// if( count>100 )
			// {
			// 	count = 0;
			// 	imshow("Closed States", closed);
			// 	imshow("Visited States", visited);
			// 	waitKey(10);	
			// }	
			// else count++;


			// cout<<front->x<<" "<<front->y<<endl;
			// cout<<front->prnt->x<<" "<<front->prnt->y<<endl;

			if( isReached(*front))
			{
				reached = true;
			}	

			// checks if the expanded state is an Island 
			// if true pop it out of list, as once expanded no more dummy edge should be created to it
			// if island parent then use normal heuristic else other wala heuristic just this is different
			bool islandChild = true; 
			if( front->islandParent == false)
			{
				if(	isIsland(*front) == false )
					islandChild = false;
			}
				
			// This for loop takes care of the usual 8 neighbours 
			for (int i = -connNeighbours/2; i <= connNeighbours/2; ++i)
				for (int j = -connNeighbours/2; j <= connNeighbours/2; ++j)
				{
					if( i==0 && j==0 ) continue; 
					
					int nextX,nextY;
					nextX = front->x + i, nextY = front->y +j;

					if( (img.at<Vec3b>(nextY,nextX) != Vec3b(255,255,255))
						||(nextX<0 || nextX>=imgCols ||  nextY<0 || nextY>=imgRows) )
						continue;

					if( islandChild == true )
					{	
						// cout<<"Entered into island search" <<endl;
						if( visited.at<Vec3b>(nextY,nextX) == Vec3b(0,0,255) )
						{
							State *next;
							next = record[nextX][nextY]; 	

							if( next->g  > front->g + getCost(*front, *next) )
							{
								next->g = front->g + getCost(*front, *next);
								next->h = getHeuristic(*next);
								next->prnt = front;	
								next->islandParent = true;


							}

						}
						else
						{
							// printf("New state found\n");
							visited.at<Vec3b>(nextY,nextX) = Vec3b(0,0,255);
							
							State *next = new State;
							next->x = nextX, next->y = nextY;
							record[next->x][next->y]  = next;

							next->g = front->g + getCost(*front, *next);
							next->h = getHeuristic(*next);
							next->prnt = front;	
							next->islandParent = true;

							pq.push(next);  
						}
					}	
					else 
					{
						if( visited.at<Vec3b>(nextY,nextX) == Vec3b(0,0,255) )
						{
							State *next;
							next = record[nextX][nextY]; 	

							if( next->g  > front->g + getCost(*front, *next) )
							{
								next->g = front->g + getCost(*front, *next);
								next->h = getHeuristicThroughIsland(*next);
								next->prnt = front;	
								next->islandParent = false;

							}

						}
						else
						{
							// printf("New state found\n");
							visited.at<Vec3b>(nextY,nextX) = Vec3b(0,0,255);
							
							State *next = new State;
							next->x = nextX, next->y = nextY;
							record[next->x][next->y]  = next;

							next->g = front->g + getCost(*front, *next);
							next->h = getHeuristicThroughIsland(*next);
							next->prnt = front;	
							next->islandParent = false;

							pq.push(next);  
						}
					}
				}
		}
	

		if( reached )
		{
			double pathLength = 0;
			State *front = record[end.x][end.y];
			while( !(front->x == start.x && front->y == start.y ))
			{
				// cout<<front->x<<" "<<front->y<<" "<<front->prnt->x<<" "<<front->prnt->y<<endl;
				pathLength += getCost(*front, *(front->prnt));
				final.at<Vec3b>(front->y, front->x) = Vec3b(255,0,0);
				front = front->prnt;
	        }  
	        cout<<"Path length: "<<pathLength<<endl;
			imwrite(destinationPath+"final_"+to_string(experimentNumber)+".png", final);
			imwrite(destinationPath+"expanded_"+to_string(experimentNumber)+".png", closed);
			waitKey(10);
			return pathLength;
		}
		else
		{
			cout<<"Path not found"<<endl;
			return 0;
		}
	
	}

	double getCost( State curr, State next)
	{
		return sqrt(pow(curr.x-next.x,2)+pow(curr.y-next.y,2));
	}


	bool isReached(State curr )
	{
		if( curr.x == end.x && curr.y == end.y )
		{
			// cout<<"Final h1 value: "<<curr.h1<<endl;
			return true;
		}	
		else
			return false;
	}

	bool isIsland(State curr )
	{
		for (int i = 0; i < islands.size(); ++i)
		{
			if( curr.x == islands[i].x && curr.y == islands[i].y)
			{
				// islands.erase(islands.begin()+i);
				// cout<<"Island is expanded with h1 value: "<<curr.h1<<" with coordinates: "<<curr.x<<" "<<curr.y<<endl;
				return true;
			}
		}
		return false;
	}



	void printPath(State curr)
	{

		if( curr.x == start.x && curr.y == start.y )
			return ;

	    // cout<<curr.prnt->x<<" "<<curr.prnt->y<<endl;
	    img.at<Vec3b>(curr.y, curr.x) = Vec3b(255,0,0);
	    printPath( *(curr.prnt) );   
          
    }

    float getHeuristic( State curr)
    {
 		return sqrt(pow(curr.x-end.x,2)+pow(curr.y-end.y,2));
    }

    float getHeuristicThroughIsland( State curr)
    {
    	float minDist = INT_MAX, currDist = 0.0;
    	for (int i = 0; i < islands.size(); ++i)
		{
			currDist = getCost(curr, islands[i]) + getCost(islands[i],end);
			if(currDist<minDist)
				minDist = currDist;
		}
 		return minDist;
     }

private:
 	int connNeighbours = 3;
    Mat img;
    int imgCols, imgRows;
    State start, end;
    vector<State> path;
    State ***record;
    vector<State> islands;
    float activationRadius = 75;
};

int main(int argc, char *argv[])
{
	int N;
	string imagePath;

	cin >>imagePath;	
	cout<<imagePath<<endl;
	cin >>destinationPath;
	cout<<destinationPath<<endl;


    // Number of instances
	cin >>N;
    // Control the weight of weighted A* 
	cin >>weight; 
	
	Mat img_gray;
	Mat img = imread(imagePath,1);
	cvtColor(img, img_gray, CV_BGR2GRAY);

	// Using Harris corner detection to identify islands 
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros( img.size(), CV_32FC1 );

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris( img_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

	/// Normalizing
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );

	vector<State> islands;
	for( int j = 0; j < dst_norm.rows ; j++ )
	    for( int i = 0; i < dst_norm.cols; i++ )
      	{
	        if( (int) dst_norm.at<float>(j,i) > 200 )
	        {
	        	State temp = {i,j};
	        	islands.push_back(temp);
	        	img.at<Vec3b>(j, i) = Vec3b(255,255,255);
				// circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
          	}
      	}	


	State start,end;
	for (experimentNumber = 0; experimentNumber < N; ++experimentNumber)
	{
		
		// do{
		// 	start.x = rand()%400;
		// 	start.y = rand()%400;
		// } while( img.at<Vec3b>(start.y, start.x) == Vec3b(0,0,0));	

		// do{
		// 	end.x = rand()%400;
		// 	end.y = rand()%400;
		// } while( img.at<Vec3b>(end.y, end.x) == Vec3b(0,0,0));	

		cin >>start.x >> start.y;
		cin >>end.x >> end.y;

		cout<<"Start: "<<start.x<<" "<<start.y<<endl;
		cout<<"End: "<<end.x<<" "<<end.y<<endl;

		// Initialise the planner and call to plan
		expanded_states = 0;
		clock_t time = clock();
		double timeTaken = 0;
		PathPlanner path(start, end, img.clone(), islands);
		path.getPath(); 
		timeTaken = (clock() - time)/(double)CLOCKS_PER_SEC;

		cout<<"Time taken: "<<timeTaken<<endl;
		cout<<"Total states expanded: "<<expanded_states<<endl;

		cout<<start.x<<" "<<start.y<<endl;
		cout<<end.x<<" "<<end.y<<endl;

	}
	return 0;
}
