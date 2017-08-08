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

// declare a random engine to be used in methods below


void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    // Set the random engine
    default_random_engine rand_engine;
    // Set number of particles
    num_particles = 100;
    // Initialize the noises
    normal_distribution<double> N_x_gaussian(0, std[0]);
    normal_distribution<double> N_y_gaussian(0, std[1]);
    normal_distribution<double> N_theta_gaussian(0, std[2]);
    // Initialize particles
    for (int i=0; i < num_particles; i++){
        // Instantiate particle object
        Particle p;
        // Set the attributes
        p.id = i;
        p.x = x + N_x_gaussian(rand_engine);
        p.y = y + N_y_gaussian(rand_engine);
        p.theta = theta + N_theta_gaussian(rand_engine);
        p.weight = 1.0;
        // Add it to the particle list, its a vector type
        particles.push_back(p);
        weights.push_back(p.weight);
    }
    // Set initialization flag
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    // Set the random engine
    default_random_engine rand_engine;
    // Initialize the noises
    normal_distribution<double> N_x_gaussian(0, std_pos[0]);
    normal_distribution<double> N_y_gaussian(0, std_pos[1]);
    normal_distribution<double> N_theta_gaussian(0, std_pos[2]);

    for (int i=0; i < num_particles; i++){
        // Account for very low yaw rate since it appears in the denominator
        // Assign the values
        if (fabs(yaw_rate) < 0.0001){
            particles[i].x += velocity*delta_t*cos(particles[i].theta);
            particles[i].y += velocity*delta_t*sin(particles[i].theta);
            // Since the yaw rate was almost zero, the theta practically remains the same
        } else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }
        // Add the noise to the values
        particles[i].x += N_x_gaussian(rand_engine);
        particles[i].y += N_y_gaussian(rand_engine);
        particles[i].theta += N_theta_gaussian(rand_engine);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0; i < observations.size(); i++){
        // Set the minimum distance to the landmark as the maximum machine value
        double dist_min = numeric_limits<double>::max();
        // Get the current observation
        LandmarkObs obs = observations[i];
        // Initialize landmark id associated with the observation
        int asso_id = -1;
        // Loop through the predictions to find the closest landmark
        for (int j = 0; j < predicted.size(); j++){
            // Get the current predicted landmark
            LandmarkObs pred = predicted[j];
            // Calculate the distance
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            // Is this the closest?
            if (distance < dist_min){
                // Change the distance and update the association id
                asso_id = pred.id;
                dist_min = distance;
            }
        }
        // Update the observation with the closest predicted landmark id
        observations[i].id = asso_id;
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

  // Varible to store the total sum of weights for normalization
  float sum_weights = 0.0;
  // Cycle through all the particles
  for (int i = 0; i < num_particles; i++) {
    // Initialize weight
    double wt = 1.0;
    // Transform the coordinates for the observations
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs obs = observations[j];
      LandmarkObs trans_obs;
      // Transform the coordinates through simple rotations and translations
      // x_new = x*CosA - y*SinA;
      // y_new = x*SinA + y*SinA;
      trans_obs.x = (obs.x * cos(particles[i].theta)) - (obs.y * sin(particles[i].theta)) + particles[i].x;
      trans_obs.y = (obs.x * sin(particles[i].theta)) + (obs.y * cos(particles[i].theta)) + particles[i].y;
      trans_obs.id = obs.id;

      LandmarkObs landmark;
      double distance_min = numeric_limits<double>::max();

      for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
        double distance = dist(trans_obs.x, trans_obs.y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
        if (distance < distance_min) {
          distance_min = distance;
          landmark = LandmarkObs{map_landmarks.landmark_list[k].id_i, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f};
        }
      }
      //Find the associated 2D gaussian value and keep on multiplying with weights
      double sig_x = std_landmark[0], sig_y = std_landmark[1];
      wt *= (1/(2*M_PI*sig_x*sig_y))*exp(-0.5 * (pow((trans_obs.x - landmark.x), 2) / pow(sig_x, 2) +
                               pow((trans_obs.y - landmark.y), 2) / pow(sig_y, 2)));
    }
    // Assign the weights
    sum_weights += wt;
    particles[i].weight = wt;
  }
  // Normalize the weights
  for (int i = 0; i < num_particles; i++) {
    particles[i].weight /= sum_weights;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Set the random engine
    default_random_engine rand_engine;
    // Form the random distribution with the weights
    discrete_distribution<int> disc_dist(weights.begin(), weights.end());
    // Prepare the temporary resampled particles vector
    vector<Particle> re_particles;
    // Resample from the distribution
    for (int i = 0; i < num_particles; i++){
        re_particles.push_back(particles[disc_dist(rand_engine)]);
    }
    // Update the particles vector
    particles = re_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

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
