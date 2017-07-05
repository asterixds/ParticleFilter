/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;
    random_device r;
    default_random_engine generator(r());
    normal_distribution<double> x_init(x, std[0]);
    normal_distribution<double> y_init(y, std[1]);
    normal_distribution<double> theta_init(theta, std[2]);
    
    for (int i=0; i< num_particles;++i){
        Particle p;
        p.id =i;
        p.x = x_init(generator);
        p.y = y_init(generator);
        p.theta = theta_init(generator);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(p.weight);
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    random_device r;
    default_random_engine generator(r());
    normal_distribution<double> x_init(0, std_pos[0]);
    normal_distribution<double> y_init(0, std_pos[1]);
    normal_distribution<double> theta_init(0, std_pos[2]);
    
    
    for (int i=0; i< particles.size();++i){
        Particle& p = particles[i];
        if(yaw_rate == 0){
            p.x += x_init(generator) + velocity*delta_t*cos(p.theta);
            p.y += y_init(generator) + velocity*delta_t*sin(p.theta);
        } else {
            p.x += x_init(generator) + (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t)- sin(p.theta));
            p.y += y_init(generator) + (velocity/yaw_rate)*(cos(p.theta)-cos(p.theta + yaw_rate*delta_t));
            p.theta += theta_init(generator) + yaw_rate*delta_t;
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    
    for (int i=0;i< observations.size();++i){
        LandmarkObs& obs = observations[i];
        double min_dist = numeric_limits<double>::max();
        for (int j = 0; j < predicted.size(); ++j) {
            LandmarkObs pred = predicted[j];
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (distance < min_dist) {
                min_dist = distance;
                obs.id = j;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    
    for (int i=0;i<particles.size();++i){
        //Apply actual measurements to each particle while transforming from vehicle to map coordinates
        Particle& particle = particles[i];
        vector<LandmarkObs> actual_observations;
        for (int j = 0; j < observations.size(); ++j) {
            LandmarkObs map_obs;
            LandmarkObs obs = observations[j];
            map_obs.x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
            map_obs.y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
            actual_observations.push_back(map_obs);
        }
        
        //generate the prediction vector. the difference between the predicted measurement
        //and the actual measurements will be used to create the importance weights for resampling
        vector<LandmarkObs> predictions;
        for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
            Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
            if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range) {
                LandmarkObs prediction = {landmark.id_i, landmark.x_f, landmark.y_f};
                predictions.push_back(prediction);
            }
        }
        
        //Find the predicted measurement that is closest to each observed measurement and assign the
        //   observed measurement to this particular landmark
        dataAssociation(predictions, actual_observations);
        
        // update the particle weights (pdf) based on squared error between landmark predictions vs actual observations
        particle.weight = 1.0;
        
        for (int l=0;l< actual_observations.size();l++){
            LandmarkObs actual = actual_observations[l];
            LandmarkObs pred = predictions[actual.id];
            
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
            
            double sqerror_x = pow(actual.x -pred.x,2 );
            double sqerror_y = pow(actual.y -pred.y,2 );
            
            double prob = exp(-((sqerror_x/2*std_x*std_x) + (sqerror_y/2*std_y*std_y))) / (2 * M_PI * std_x * std_y);
            particle.weight *= prob;
            
        }
        weights[i] = particle.weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    random_device r;
    default_random_engine generator(r());
    discrete_distribution<double> dist_weights(weights.begin(), weights.end());
    
    vector<Particle> resampled_particles;
    for (int i = 0; i < particles.size(); ++i) {
        resampled_particles.push_back(particles[dist_weights(generator)]);
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    
    
    particle.associations.clear();
    
    particle.sense_y.clear();
    particle.sense_x.clear();
    
    particle.associations= associations;
    particle.sense_y = sense_y;
    particle.sense_x = sense_x;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
