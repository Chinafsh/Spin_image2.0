#ifndef ORIENTEDPOINT_H
#define ORIENTEDPOINT_H

#include<OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include<opencv2/core.hpp>


class orientedPoint{
    public:
        OpenMesh::Vec3f Position;
        OpenMesh::Vec3f Normal;
        cv::Mat* SpinImage; // Spin image
       
        orientedPoint()
        {
            Position = OpenMesh::Vec3f(.0f);
            Normal = OpenMesh::Vec3f(.0f);
        }   
    
};

#endif