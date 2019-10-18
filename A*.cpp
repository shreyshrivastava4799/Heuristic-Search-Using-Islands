#include <queue>
#include <iostream>
#include <bits/stdc++.h>
#include <unistd.h>
#include <ctime>

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
	double g1;
	double h1;
	double h;
	struct _State *prnt;


} State;

typedef struct _Island
{
	int x;
	int y;
}Island;

static float weight = 1;
int expanded_states = 0, visited_states = 0;


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
		return a->g1 + a->h1 + weight*a->h > b->g1 + b->h1 + weight*b->h;
	}
};


class PathPlanner
{
		
 
public:

	PathPlanner(State start, State end, Mat img, vector<Island> islands)
	{
        cout<<"Planner Initiated"<<endl;
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
	   	cout<<"Inside Getpath"<<endl;			
		bool reached = false;

		Mat closed = img.clone();
		Mat visited = img.clone();

		// Show start and end points
		circle(img, Point(start.x, start.y), 2, Vec3b(255,0,0), 1);
		circle(img, Point(end.x, end.y), 2, Vec3b(0,255,0), 1);
		imshow("Planning Problem", img);
		waitKey(0);

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
		start.g1 = 0;
		start.h1 = 0;
		start.h = getHeuristic(start);
		start.prnt = NULL;
		pq.push(&start);
		record[start.x][start.y] = &start;
		visited.at<Vec3b>(start.y,start.x) = Vec3b(0,0,255);
		

		while(!pq.empty() && !reached )
		{
			
			State *front = pq.top();
			pq.pop();
			expanded_states++;
			
			// Visualisation of visited and closed states 
			closed.at<Vec3b>(front->y,front->x) = Vec3b(0,255,0);
			imshow("Closed States", closed);
			imshow("Visited States", visited);
			waitKey(10);	

			// cout<<front->x<<" "<<front->y<<endl;
			// cout<<front->prnt->x<<" "<<front->prnt->y<<endl;

			if( isReached(*front))
			{
				printf("***********************Reached**************************\n");
				reached = true;
			}	

			// checks if the expanded state is an Island 
			// if true pop it out of list, as once expanded no more dummy edge should be created to it
			isIsland(*front);

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

					visited_states++;
					if( visited.at<Vec3b>(nextY,nextX) == Vec3b(0,0,255) )
					{
						State *next;
						next = record[nextX][nextY]; 	

						if( next->g1 + next->h1  > front->g1 + front->h1 + getCost(*front, *next) )
						{
							next->g1 = front->g1 + getCost(*front, *next);
							next->h1 = front->h1;

							next->h = getHeuristic(*next);
							next->prnt = front;	

						}

					}
					else
					{
						// printf("New state found\n");
						visited.at<Vec3b>(nextY,nextX) = Vec3b(0,0,255);
						
						State *next = new State;
						next->x = nextX, next->y = nextY;
						record[next->x][next->y]  = next;

						next->g1 = front->g1 + getCost(*front, *next);
						next->h1 = front->h1;

						next->h = getHeuristic(*next);
						next->prnt = front;	

						pq.push(next);  
					}

				}
			
						
			// This will take care of the island successors 
			for (int i = 0; i < islands.size(); ++i)
			{

				// checking if inside activation region
				double islandDist = sqrt(pow(islands[i].x-front->x,2)+pow(islands[i].y-front->y,2));
				if(  islandDist > activationRadius || islandDist == 0 )
					continue ;

				// these islands are the expanded nodes or neighbours for current nodes	
				int nextX,nextY;
				nextX = islands[i].x, nextY = islands[i].y;


				if( (img.at<Vec3b>(nextY,nextX) != Vec3b(255,255,255))
					||(nextX<0 || nextX>=imgCols ||  nextY<0 || nextY>=imgRows) )
					continue;


				visited_states++;
				if( visited.at<Vec3b>(nextY,nextX) == Vec3b(0,0,255) )
				{
					State *next;
					next = record[nextX][nextY]; 	

					if( next->g1 + next->h1  > front->g1 + front->h1 + getCost(*front, *next) )
					{
						next->g1 = front->g1 ;
						next->h1 = front->h1 + getCost(*front, *next);

						next->h = getHeuristic(*next);
						next->prnt = front;	
						
					}

				}
				else
				{
					// printf("New state found\n");
					visited.at<Vec3b>(nextY,nextX) = Vec3b(0,0,255);
					
					State *next = new State;
					next->x = nextX, next->y = nextY;
					record[next->x][next->y]  = next;

					next->g1 = front->g1 ;
					next->h1 = front->h1 + getCost(*front, *next);

					next->h = getHeuristic(*next);
					next->prnt = front;	

					// cout<<"Island is pushed with coordinates: "<<nextX<<" "<<nextY<<" and h1: "<<next->h1<<endl;
					pq.push(next);  
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
				img.at<Vec3b>(front->y, front->x) = Vec3b(255,0,0);
				front = front->prnt;
	        }  
	        cout<<"The length of path found including dummy edges: "<<pathLength<<endl;
			imshow("Path generated", img);
			waitKey(0);

			// if dummy edges have been used remove them 
			if( (*record[end.x][end.y]).h1 != 0 )
			{
				cout<<"removeDummyEdges() called "<<endl;
			 	pathLength = 0;
				vector<Island> v;
				State *front = record[end.x][end.y];
				while( !(front->x == start.x && front->y == start.y ))
				{
					if( getCost(*front, *(front->prnt)) > 2 )
					{
						img.at<Vec3b>(front->y, front->x) = Vec3b(255,255,255);
						img.at<Vec3b>((front->prnt)->y, (front->prnt)->x) = Vec3b(255,255,255);
						
						PathPlanner islandPath(*(front->prnt), *front, img, v);
						pathLength += islandPath.getPath();
						
						imshow("Path generated", img);
						waitKey(0);
					}	
					else
						pathLength += getCost(*front, *(front->prnt));
					front = front->prnt;
			    } 
			}
			
	        cout<<"The length of path found including dummy edges: "<<pathLength<<endl;
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

	void isIsland(State curr )
	{
		for (int i = 0; i < islands.size(); ++i)
		{
			if( curr.x == islands[i].x && curr.y == islands[i].y)
			{
				islands.erase(islands.begin()+i);
				// cout<<"Island is expanded with h1 value: "<<curr.h1<<" with coordinates: "<<curr.x<<" "<<curr.y<<endl;
				return;
			}
		}
	}



	void printPath(State curr)
	{

		if( curr.x == start.x && curr.y == start.y )
			return ;

	    img.at<Vec3b>(curr.y, curr.x) = Vec3b(255,0,0);
	    // cout<<curr.prnt->x<<" "<<curr.prnt->y<<endl;
	    printPath( *(curr.prnt) );   
          
    }

    float getHeuristic( State curr)
    {
 		return sqrt(pow(curr.x-end.x,2)+pow(curr.y-end.y,2));
    }


private:
 	int connNeighbours = 3;
    Mat img;
    int imgCols, imgRows;
    State start, end;
    vector<State> path;
    State ***record;
    vector<Island> islands;
    float activationRadius = 75;
};

int main(int argc, char *argv[])
{

	string imagePath;
	cin >>imagePath;	
	cout<<imagePath<<endl;
	
	Mat img = imread(imagePath,1);
	Mat img_gray;
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

	vector<Island> islands;
	/// Drawing a circle around corners
	for( int j = 0; j < dst_norm.rows ; j++ )
	    for( int i = 0; i < dst_norm.cols; i++ )
      	{
	        if( (int) dst_norm.at<float>(j,i) > 200 )
	        {
	        	Island temp = {i,j};
	        	islands.push_back(temp);
	        	img.at<Vec3b>(j, i) = Vec3b(255,255,255);
				// circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
          	}
      	}	

	// /// Showing the result
	// for (int i = 0; i < islands.size(); ++i)
	// 	cout<<islands[i].x<<" "<<islands[i].y<<endl;
	// imshow( "Corners Window", dst_norm_scaled );
	// waitKey(0);

    // Control the weight of weighted A* 
	cin >>weight; 

	// Starting State and ending State
	State start, end;  
	cout<<"Give start x and start y"<<endl; 
	cin >>start.x >>start.y; 
	cout<<start.x<<" "<<start.y<<endl;
	cout<<"Give end x and end y"<<endl; 
	cin >>end.x >>end.y; 
	cout<<end.x<<" "<<end.y<<endl;;

	// Initialise the planner and call to plan
	clock_t time = clock();
	double timeTaken = 0;
	PathPlanner path(start, end, img, islands);
	path.getPath(); 
	timeTaken = (clock() - time)/(double)CLOCKS_PER_SEC;

	cout<<"Total states visited: "<<visited_states<<endl;
	cout<<"Total states expanded: "<<expanded_states<<endl;
	cout<<"Time taken: "<<timeTaken<<endl;
	return 0;
}
