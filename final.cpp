#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <iomanip>
#include <limits>
#include <string>
#include <algorithm>

//generating random coordinates within 0.0-1.0 boundaries
void generate_random_coordinates(std::vector<std::pair<double, double>>& coordinates, size_t n_points) 
{
    std::random_device rd; //random generator
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); //uniform distribution between the given boundaries

    coordinates.reserve(n_points); //reserving space for future efficiency
    
    for (size_t i = 0; i < n_points; ++i) //generating random coordinates
    {
        coordinates.emplace_back(dis(gen), dis(gen)); //add the generated x and y coordinates to the defined vector
    }
}

//reading coordinates from CSV files
void read_coordinates_from_csv(const std::string& filename, std::vector<std::pair<double, double>>& coordinates)
{
    std::ifstream file(filename);
    
    ///// test if the files can be read
    // if (!file.is_open()) 
    // {
    //     std::cerr << "Error opening file: " << filename << std::endl; 
    //     return;
    // }
    
    std::string line; //reading each line of the file
    
    while (std::getline(file, line)) 
    {
        size_t comma_pos = line.find(','); //finding the comma separator
        
        if (comma_pos == std::string::npos) continue; //skipping the lines without the comma
        double x = std::stod(line.substr(0, comma_pos)); //extract the x coordinates
        double y = std::stod(line.substr(comma_pos + 1)); //extract the y coordinates
        coordinates.emplace_back(x, y); //add the pairs to the vector
    }
    
    file.close(); //closing the stream
}

//calculating Euclidean (standard) distance 
double euclidean_distance(const std::pair<double, double>& a, const std::pair<double, double>& b) 
{
    double dx = a.first - b.first; //difference in x coordinates
    double dy = a.second - b.second; //difference in y coordinates
    return std::sqrt(dx * dx + dy * dy); //using the pythagorean theorem (Equation 1 in report) to find the eucledian distance
}

//calculating wraparound distance
double wraparound_distance(const std::pair<double, double>& a, const std::pair<double, double>& b) 
{
    double dx = std::abs(a.first - b.first); //absolute difference in x coordinates
    double dy = std::abs(a.second - b.second); //absolute difference in y coordinates
    dx = std::min(dx, 1.0 - dx); //wrapping effect for x coordinates(Equation 2 in report)
    dy = std::min(dy, 1.0 - dy); //wrapping effect for y coordinates(Equation 2 in report)
    return std::sqrt(dx * dx + dy * dy); //calculating the distance as shown in Equation 2
}

//computing nearest and furthest distances for each point
void compute_distances(const std::vector<std::pair<double, double>>& coordinates, 
                       std::vector<double>& nearest_distances, 
                       std::vector<double>& furthest_distances, 
                       bool wraparound, 
                       double& sum_nearest, 
                       double& sum_furthest)
{
    size_t n = coordinates.size();
    sum_nearest = 0.0; //initialisng the sum for the nearest distances
    sum_furthest = 0.0; //initialisng the sum for the furthest distances

    #pragma omp parallel for schedule(dynamic) reduction(+:sum_nearest,sum_furthest) //applying parallelisation
    
    for (size_t i = 0; i < n; ++i) 
    {
        double nearest = std::numeric_limits<double>::max(); //initiallising the nearest distance to the highest possible value
        double furthest = 0.0; //initilaising furthest distance to 0
        
        for (size_t j = 0; j < n; ++j) //looping over all points 
        {
            if (i == j) continue; //skipping the same point
            double dist = wraparound ? wraparound_distance(coordinates[i], coordinates[j])
                                     : euclidean_distance(coordinates[i], coordinates[j]); //calculating the distance based on the geometry
            
            if (dist < nearest) nearest = dist; //updating the nearest distances
            
            if (dist > furthest) furthest = dist; //updating the furthest distances
        }

        nearest_distances[i] = nearest; //storing the nearest distances
        furthest_distances[i] = furthest; //storing the furthest distances
        
        sum_nearest += nearest; //updating the sums for calculating the average values of the nearest distances
        sum_furthest += furthest; //updating the sums for calculating the average values of the furthest distances
    }
}


//processing datasets and computing the distances
void process_data_set(const std::vector<std::pair<double, double>>& coordinates, const std::string& output_prefix)
{
    size_t n = coordinates.size();
   
    std::vector<double> nearest_distances(n); //vectors for storing the nearest distances
    std::vector<double> furthest_distances(n); //vectors for storing the furthest distances

    double sum_nearest_standard = 0.0; //initialsing the sums for both geometries
    double sum_furthest_standard = 0.0;
    double sum_nearest_wraparound = 0.0;
    double sum_furthest_wraparound = 0.0;
    
    double avg_nearest_standard = 0.0; //initialsing the averages for both geometries
    double avg_furthest_standard = 0.0;
    double avg_nearest_wraparound = 0.0;
    double avg_furthest_wraparound = 0.0;

    //computing distances (standard geometry)
    double start_time = omp_get_wtime(); //recording the start time
    compute_distances(coordinates, nearest_distances, furthest_distances, false, sum_nearest_standard, sum_furthest_standard);
    double end_time = omp_get_wtime(); //recording the end time
    double elapsed_time_standard = end_time - start_time; //calculating the passed time

    //calculating the average distances for standard geometry
    avg_nearest_standard = sum_nearest_standard / n;
    avg_furthest_standard = sum_furthest_standard / n;

    //nearest distances (standard geometry)
    std::ofstream nearest_file(output_prefix + "_nearest_distances_standard.txt");
    for (double dist : nearest_distances) 
    {
        nearest_file << dist << "\n";
    }
    nearest_file.close();

    //furthest distances (standard geometry)
    std::ofstream furthest_file(output_prefix + "_furthest_distances_standard.txt");
    for (double dist : furthest_distances) 
    {
        furthest_file << dist << "\n";
    }
    furthest_file.close();

    //average distances output for standard geometry
    std::cout << "Standard Geometry (" << output_prefix << "):" << std::endl;
    std::cout << "Average nearest distance: " << avg_nearest_standard << std::endl;
    std::cout << "Average furthest distance: " << avg_furthest_standard << std::endl;
    std::cout << "Time taken: " << elapsed_time_standard << " seconds." << std::endl;

    //computing distances (wraparound geometry)
    start_time = omp_get_wtime(); //recording start time
    compute_distances(coordinates, nearest_distances, furthest_distances, true, sum_nearest_wraparound, sum_furthest_wraparound);
    end_time = omp_get_wtime(); //recording end time
    double elapsed_time_wraparound = end_time - start_time; //calculating the passed time

    //calculating the average distances for wraparound geometry
    avg_nearest_wraparound = sum_nearest_wraparound / n;
    avg_furthest_wraparound = sum_furthest_wraparound / n;

    //nearest distances (wraparound geometry)
    nearest_file.open(output_prefix + "_nearest_distances_wraparound.txt");
    for (double dist : nearest_distances) 
    {
        nearest_file << dist << "\n";
    }
    nearest_file.close();

    //furthest distances (wraparound geometry)
    furthest_file.open(output_prefix + "_furthest_distances_wraparound.txt");
    for (double dist : furthest_distances) 
    {
        furthest_file << dist << "\n";
    }
    furthest_file.close();

    //average distances output for wraparound geometry
    std::cout << "Wraparound Geometry (" << output_prefix << "):" << std::endl;
    std::cout << "Average nearest distance: " << avg_nearest_wraparound << std::endl;
    std::cout << "Average furthest distance: " << avg_furthest_wraparound << std::endl;
    std::cout << "Time taken: " << elapsed_time_wraparound << " seconds." << std::endl;

    //when the processing is complete, display:
    std::cout << "Processing complete for " << output_prefix << ". Check output files for results." << std::endl << std::endl;
}


//main function
int main(int argc, char* argv[]) 
{
    size_t n_points = 100000;     //default number of points, 100k

    //variables for controlling the OpenMP settings
    int num_threads = omp_get_max_threads();
    omp_sched_t schedule_kind = omp_sched_static;
    int chunk_size = 0;


    ///// command-line arguments using: program_name -t num_threads -s schedule_kind -c chunk_size /////


    //argument for parse command line
    for (int i = 1; i < argc; ++i) 
    {
        std::string arg = argv[i];
       
        if (arg == "-t" && i + 1 < argc) 
        {
            num_threads = std::stoi(argv[++i]); //number of threads
        } 
        else if (arg == "-s" && i + 1 < argc) 
        {    
            std::string sched = argv[++i]; //type of scheduling
            if (sched == "static") 
            {
                schedule_kind = omp_sched_static;
            } 
            else if (sched == "dynamic") 
            {
                schedule_kind = omp_sched_dynamic;
            } 
            else if (sched == "guided") 
            {
                schedule_kind = omp_sched_guided;
            } 
            else if (sched == "auto") 
            {
                schedule_kind = omp_sched_auto;
            } 
            else 
            {
                std::cerr << "Unknown schedule type: " << sched << std::endl; //for unknown type of scheduling
                return 1;
            }
        }
        else if (arg == "-c" && i + 1 < argc) 
        {
            chunk_size = std::stoi(argv[++i]); //chunk size
        }
        else if (arg == "-n" && i + 1 < argc) 
        {
            n_points = std::stoul(argv[++i]); //number of points
        }
        else 
        {
            std::cerr << "Unknown argument: " << arg << std::endl; //for unknown arguments
            return 1;
        }
    }

    //defining the output for the OpenMP settings
    omp_set_num_threads(num_threads);
    omp_set_schedule(schedule_kind, chunk_size);
    std::cout << "Using " << num_threads << " threads." << std::endl;
    std::cout << "Schedule kind: ";
    switch (schedule_kind) 
    {
        case omp_sched_static:
            std::cout << "static";
            break;
        case omp_sched_dynamic:
            std::cout << "dynamic";
            break;
        case omp_sched_guided:
            std::cout << "guided";
            break;
        case omp_sched_auto:
            std::cout << "auto";
            break;
        default:
            std::cout << "unknown";
    }
    std::cout << " with chunk size " << chunk_size << "." << std::endl << std::endl;

    //processing randomly generated coordinates
    std::vector<std::pair<double, double>> coordinates; //setting  a vector to store the coordinates
    std::cout << "Processing randomly generated coordinates (" << n_points << " points):" << std::endl;
    
    //processing the randomly generated coordinates
    generate_random_coordinates(coordinates, n_points);
    process_data_set(coordinates, "random_" + std::to_string(n_points));
    coordinates.clear(); //empty the vector for clearing the memory

    //processing coordinates from "100000 locations.csv"
    coordinates.clear();
    std::cout << "Processing coordinates from '100000 locations.csv':" << std::endl;
    read_coordinates_from_csv("C:\\Users\\marij\\OneDrive\\Desktop\\100000 locations.csv", coordinates); //reading the coordinates
    if (!coordinates.empty())
    {
        process_data_set(coordinates, "100000_locations");
    }
    else
    {
        std::cerr << "Failed to read coordinates from '100000 locations.csv'." << std::endl; //in case there is an error in reading the sets
    }
    coordinates.clear();

    //processing coordinates from "200000 locations.csv"
    coordinates.clear();
    std::cout << "Processing coordinates from '200000 locations.csv':" << std::endl;
    read_coordinates_from_csv("C:\\Users\\marij\\OneDrive\\Desktop\\200000 locations.csv", coordinates); //reading the coordiantes
    if (!coordinates.empty())
    {
        process_data_set(coordinates, "200000_locations");
    }
    else
    {
        std::cerr << "Failed to read coordinates from '200000 locations.csv'." << std::endl; //in case there is an error in reading the sets
    }
    coordinates.clear(); //clearing the vector

    return 0;
}