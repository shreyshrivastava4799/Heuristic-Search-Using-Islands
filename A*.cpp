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
	double g;
	double h;
	struct _State *prnt;

} State;

float weight = 1;

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
		return a->g + a->h > b->g + b->h;
	}
};


class PathPlanner
{
	
 
public:

	PathPlanner(State start, State end, Mat img)
	{
        cout<<"Planner Initiated"<<endl;

		this->start = start;
		this->end = end;
		this->img = img;
		imgCols = img.cols;
		imgRows = img.rows;

	}

	void getPath()
	{
	   	cout<<"Inside Getpath"<<endl;	
		
		bool reached = false;
		unsigned int microseconds = pow(10,3);

		Mat closed = img.clone();
		Mat visited = img.clone();

		// Show start and end points
		circle(img, Point(start.x, start.y), 2, Vec3b(255,0,0), 1);
		circle(img, Point(end.x, end.y), 2, Vec3b(0,255,0), 1);
		imshow("Planning Problem", img);
		waitKey(0);

		record = new State**[imgRows];
		for (int i = 0; i < imgRows; ++i)
			record[i] = new State*[imgCols];
		
		for (int i = 0; i < imgRows; ++i)
			for (int j = 0; j < imgCols; ++j)
				record[i][j] = NULL;

		priority_queue< State*, vector<State*>, stateComparator > pq;
		start.g = 0;
		start.h = getHeuristic(start);
		start.prnt = NULL;
		
		pq.push(&start);
		record[start.x][start.y] = &start;
		visited.at<Vec3b>(start.x,start.y) = Vec3b(0,0,255);
		
		while(!pq.empty() && !reached )
		{
			
			State *front = pq.top();
			pq.pop();
			closed.at<Vec3b>(front->x,front->y) = Vec3b(0,255,0);


			// cout<<front->x<<" "<<front->y<<endl;
			// cout<<front->prnt->x<<" "<<front->prnt->y<<endl;

			if( isReached(*front))
			{
				printf("***********************Reached**************************\n");
				reached = true;
			}	


			for (int i = -connNeighbours/2; i <= connNeighbours/2; ++i)
				for (int j = -connNeighbours/2; j <= connNeighbours/2; ++j)
				{
					if( i==0 && j==0 ) continue; 
					
					int nextX,nextY;
					nextX = front->x + i, nextY = front->y +j;

					if( (img.at<Vec3b>(nextX,nextY) != Vec3b(255,255,255))
						||(nextX<0 || nextX>=imgRows ||  nextY<0 || nextY>=imgCols) )
						continue;

					if( visited.at<Vec3b>(nextX,nextY) == Vec3b(0,0,255) )
					{
						printf("Already visited\n");
						State *next;
						printf("Before extracting record\n");
						next = record[nextX][nextY]; 	
						printf("After extracting record\n");

						if( next->g > front->g + getCost(*front, *next) )
						{
							next->g = front->g + getCost(*front, *next);
							next->h = getHeuristic(*next);
							next->prnt = front;	
							
							if( closed.at<Vec3b>(next->x,next->y) == Vec3b(0,255,0))
							{
								// printf("Inconsistent states corrected\n");
								pq.push(next);
							}
						}

					}
					else
					{
						printf("New state found\n");
						visited.at<Vec3b>(nextX,nextY) = Vec3b(0,0,255);
						
						State *next = new State;
						next->x = nextX, next->y = nextY;
						record[next->x][next->y]  = next;

						next->g = front->g + getCost(*front, *next);
						next->h = getHeuristic(*next);
						next->prnt = front;	

						pq.push(next);  
					}

				}
			imshow("Visited States", visited);
			waitKey(10);	

		}
	

		

		if( reached )
		{
			int pathLength = 0;
			State *front = record[end.x][end.y];
			while( !(front->x == start.x && front->y == start.y ))
			{
				// cout<<front->x<<" "<<front->y<<endl;
				// cout<<front->prnt->x<<" "<<front->prnt->y<<endl;
				pathLength++;
				img.at<Vec3b>(front->x, front->y) = Vec3b(255,0,0);
				front = front->prnt;
	        }  
	        cout<<"The length of path found: "<<pathLength<<endl;
			imshow("Path generated", img);
			waitKey(0);
		}
		else
		{
			cout<<"Path not found"<<endl;
		}

		for (int i = 0; i < imgRows; ++i)
			for (int j = 0; j < imgCols; ++j)
			{
				if( record[i][j]!=NULL && !(i==start.x && j==start.y) && !(i==end.x && j==end.y) )
					delete record[i][j];	
			}

		cout<<"Delete Successful"<<endl;
	}

	double getCost( State curr, State next)
	{
		return sqrt(pow(curr.x-next.x,2)+pow(curr.y-next.y,2));
	}

	bool isReached(State curr )
	{
		if( curr.x == end.x && curr.y == end.y )
			return true;
		else
			return false;
	}


	void printPath(State curr)
	{

		if( curr.x == start.x && curr.y == start.y )
			return ;

	    img.at<Vec3b>(curr.x, curr.y) = Vec3b(255,0,0);
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

};

int main(int argc, char *argv[])
{
	clock_t time = clock();

	string imagePath;
	cin >>imagePath;	
	cout<<imagePath<<endl;
	
	Mat img = imread(imagePath,1);

	cin >>weight; 
	// Starting State and ending State
	State start, end;  
	cout<<"Give start x and start y"<<endl; 
	cin >>start.x >>start.y; 
	cout<<"Give end x and end y"<<endl; 
	cin >>end.x >>end.y; 

	PathPlanner path(start, end, img);
	path.getPath(); 

	time = clock() - time;
	cout<<"Time taken: "<<time<<endl;
	return 0;
}