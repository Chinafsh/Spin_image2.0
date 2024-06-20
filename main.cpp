#include<iostream>
#include <cfloat> // for max, min of float 
#include<string>
#include <chrono>   // time consumption
#include<OpenMesh/Core/IO/MeshIO.hh>
#include<OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include<opencv2/opencv.hpp>
#include "orientedPoint.h"
#include<tbb/tbb.h>
// define your type MyMesh
typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

float alpha_max, beta_max, binSize;

using namespace std;
// gargoyle100K, bunny
const string objName = "../model/gargoyle100K.obj";

float computeMeshResolution(MyMesh& mesh)
{   
    float sum = .0f;
    // iterate over all edges
    for(MyMesh::EdgeIter e_it = mesh.edges_begin(); e_it!=mesh.edges_end(); ++e_it){
        MyMesh::EdgeHandle eh = *e_it;
        
        MyMesh::HalfedgeHandle heh = mesh.halfedge_handle(eh);
        MyMesh::VertexHandle vh0 = mesh.from_vertex_handle(heh);
        MyMesh::VertexHandle vh1 = mesh.to_vertex_handle(heh);
        MyMesh::Point p0 = mesh.point(vh0);
        MyMesh::Point p1 = mesh.point(vh1);
        
        sum += (p0-p1).norm();
    }
    return sum/mesh.n_edges();
}


void SpinMap(OpenMesh::Vec3f& x, OpenMesh::Vec3f& p, OpenMesh::Vec3f& n, float& alpha, float& beta)
{
    beta = n.dot(x-p);
    alpha = std::sqrt(std::pow((x-p).norm(),2)-beta*beta);
}


void computeMaxAB(MyMesh& mesh, float& alpha_max, float& beta_max)
{
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, mesh.n_vertices()),
        [&mesh, &alpha_max, &beta_max](const tbb::blocked_range<size_t>& r){
            for(size_t i=r.begin(); i!=r.end();++i){
                // Inner loop
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(i+1,mesh.n_vertices()),
                    [&mesh, i, &alpha_max, &beta_max](const tbb::blocked_range<size_t>& r_inner){
                        for(size_t j = r_inner.begin(); j!=r_inner.end(); ++j){
                            MyMesh::VertexHandle vh = *(mesh.vertices_begin()+i);
                            MyMesh::VertexHandle vvh = *(mesh.vertices_begin()+j); 
                            MyMesh::Point orientedPointNormal = mesh.normal(vh);
                            float alpha, beta;
                            SpinMap(mesh.point(vvh),mesh.point(vh),orientedPointNormal,alpha, beta);
                            alpha_max = std::max(alpha_max, alpha);
                            beta_max = std::max(beta_max, fabs(beta));
                        }
                    }
                
                );
            }
        }    
    );



    // for(MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it!=mesh.vertices_end(); ++v_it)
    // {
    //     for(MyMesh::VertexIter vv_it = v_it+1; vv_it!=mesh.vertices_end(); ++vv_it){
    //         MyMesh::VertexHandle vh = *v_it;        // oriented_point
    //         MyMesh::VertexHandle vvh = *vv_it;      // to
    //         // Spin Map(Projection: from R^3 to R^2)
    //         MyMesh::Point orientedPointNormal = mesh.normal(vh);
    //         float alpha, beta;
    //         SpinMap(mesh.point(vvh),mesh.point(vh),orientedPointNormal,alpha, beta);
    //         alpha_max = std::max(alpha_max, alpha);
    //         beta_max = std::max(beta_max, fabs(beta));
    //     }
    // }
}


void SpinImageBin(float& alpha, float& beta, int& x, int& y)
{
    y = std::floor((beta_max-beta)/binSize);        // row
    x = std::floor(alpha/binSize);                  // col
    // std::cout << "Debug ij: ("<< y << ", " << x << ")" << std::endl; 

}


void BilinearWeight(float& alpha, float& beta, int& x, int& y, float& a, float& b)
{
    a = alpha - x*binSize;
    b = beta - y*binSize;
}

void computeSpinImge(orientedPoint& O, std::vector<orientedPoint>& Os)
{
    int contributors = 0;
    int rows = O.SpinImage->rows;
    int cols = O.SpinImage->cols;

    for(int i=0; i<Os.size(); ++i){
        float angle = O.Normal.dot(Os[i].Normal);
        if(angle > 0&&angle<M_PI/3){
            ++contributors;

            float alpha, beta;
            SpinMap(Os[i].Position, O.Position, O.Normal, alpha,beta);    // get alpha, beta

            int x, y;
            SpinImageBin(alpha, beta, x, y);        // get x, y
            
            float a, b;
            BilinearWeight(alpha, beta, x, y, a, b);    // get a, b

            if(y>=0 && y<rows && x>=0 && x < cols)
                O.SpinImage->at<float>(y, x) += (1-a)*(1-b);

            if(y+1>=0 && y+1<rows && x>=0 && x < cols)
                O.SpinImage->at<float>(y+1, x) += a*(1-b);

            if(y>=0 && y<rows && x+1>=0 && x+1 < cols)
                O.SpinImage->at<float>(y, x+1) += (1-a)*b;

            if(y+1>=0 && y+1<rows && x+1>=0 && x+1 < cols)
                O.SpinImage->at<float>(y, x) += a*b;
        }
    }
    // std::cout << "Contributors = " << contributors << std::endl;
}



int main()
{

    MyMesh mesh;
    // auto start = std::chrono::system_clock::now();

    if(!OpenMesh::IO::read_mesh(mesh, objName)){
        cerr << "Error: cannot read mesh from file" << endl;
        return 1;
    }

    mesh.request_face_normals();
    mesh.update_face_normals();
    mesh.request_vertex_normals(); 
    mesh.update_vertex_normals();  
    

    
    // auto stop = std::chrono::system_clock::now();

    // std::cout << "Read complete: \n";
    // std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    // std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    // std::cout << "          : " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " seconds\n";
    // std::cout << "          : " <<(stop - start).count() << "\n";

    // debug
    cout << "Vertices: "<< mesh.n_vertices() << endl;
    cout << "Faces:" << mesh.n_faces() << endl;

    float meshResolution = computeMeshResolution(mesh);
    binSize =  1.0 * meshResolution;
    std::cout << binSize << std::endl;

    alpha_max = FLT_MIN;
    beta_max = FLT_MIN;
    
    // compute the max alpha and max beta, then get the image size
    computeMaxAB(mesh, alpha_max, beta_max);

    std::cout << alpha_max << ", " << beta_max << std::endl;

    int rows = (2*beta_max)/binSize + 1;
    int cols = alpha_max/binSize + 1;

    std::cout << rows << ", " << cols << std::endl;

    std::vector<orientedPoint> Oriented_points(mesh.n_vertices());
    std::cout << Oriented_points.size() << std::endl;
    // iterate over all vertices
    for(int i=0; i<Oriented_points.size();++i){
        Oriented_points[i].Position = OpenMesh::Vec3f(0.f);
        Oriented_points[i].Normal = OpenMesh::Vec3f(0.f);
        Oriented_points[i].SpinImage = new cv::Mat(rows, cols, CV_32F, 0.f);
    }


    size_t index = 0;
    for(MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it!=mesh.vertices_end(); ++v_it)
    {
        MyMesh::VertexHandle vh = *v_it;
        Oriented_points[index].Position = mesh.point(vh);
        Oriented_points[index].Normal = mesh.normal(vh);
        index++;
    }

    // Compute Spin image for each oriented_point
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, Oriented_points.size()),
        [&Oriented_points](const tbb::blocked_range<size_t>& r){
            for(size_t i = r.begin(); i!=r.end();++i){
                computeSpinImge(Oriented_points[i], Oriented_points);
                std::string filename = "gargoyleSpinImages/si" + to_string(i) + ".jpg";
                *(Oriented_points[i].SpinImage) *= 255.0;
                cv::imwrite(filename,*(Oriented_points[i].SpinImage));
                delete Oriented_points[i].SpinImage;
            }
        }
    );



    // for(int i = 0; i < Oriented_points.size(); i++){
    //     computeSpinImge(Oriented_points[i], Oriented_points);
    //     std::string filename = "gargoyleSpinImages/si" + to_string(i) + ".jpg";
    //     *(Oriented_points[i].SpinImage) *= 255.0;
    //     cv::imwrite(filename,*(Oriented_points[i].SpinImage));
    //     delete Oriented_points[i].SpinImage;

    //     // cv::namedWindow("Spin Image",cv::WINDOW_NORMAL);
    //     // cv::imshow("Spin Image",*(Oriented_points[i].SpinImage));
    //     // cv::waitKey(1000);
    //     // cv::normalize(*(Oriented_points[i].SpinImage), *(Oriented_points[i].SpinImage), 0, 1, cv::NORM_MINMAX);
    // }
        std::cout << "done." << std::endl;
    return 0;
}