#ifndef __TIMER_H__
#define __TIMER_H__

#ifdef _WIN32
#include <windows.h>
#undef small
#undef large
#undef min
#undef max
#else
#include <sys/time.h>
#endif

class Timer
{
public:
	Timer()
	{
	#ifdef WIN32
		QueryPerformanceFrequency(&frequency);
		startCount.QuadPart = 0;
		endCount.QuadPart = 0;
	#else
		startCount.tv_sec = startCount.tv_usec = 0;
		endCount.tv_sec = endCount.tv_usec = 0;
	#endif
		stopped = false;
		startTimeInMicroSec = 0;
		endTimeInMicroSec = 0;
	}

	~Timer(){};                                  

	void start()
	{
		stopped = false;
	#ifdef WIN32
		QueryPerformanceCounter(&startCount);
	#else
		gettimeofday(&startCount, NULL);
	#endif		
	}

	void stop()                             
	{
		stopped = true;
	#ifdef WIN32
		QueryPerformanceCounter(&endCount);
	#else
		gettimeofday(&endCount, NULL);
	#endif
	}

	double getElapsedTimeInMicroSec()
	{
	#ifdef WIN32
		if(!stopped)
			QueryPerformanceCounter(&endCount);

		startTimeInMicroSec = startCount.QuadPart *(1000000.0 / frequency.QuadPart);
		endTimeInMicroSec = endCount.QuadPart *(1000000.0 / frequency.QuadPart);
	#else
		if(!stopped)
			gettimeofday(&endCount, NULL);

		startTimeInMicroSec =(startCount.tv_sec * 1000000.0) + startCount.tv_usec;
		endTimeInMicroSec =(endCount.tv_sec * 1000000.0) + endCount.tv_usec;
	#endif

		return endTimeInMicroSec - startTimeInMicroSec;
	}

	double getElapsedTimeInSec()      { return this->getElapsedTimeInMicroSec() * 0.000001; } 
	double getElapsedTimeInMilliSec() { return this->getElapsedTimeInMicroSec() * 0.001;    }
private:
    double startTimeInMicroSec;
    double endTimeInMicroSec;  
    bool stopped;            
#ifdef WIN32
    LARGE_INTEGER frequency;   
    LARGE_INTEGER startCount;  
    LARGE_INTEGER endCount;    
#else
    timeval startCount;        
    timeval endCount;          
#endif
};

#endif
