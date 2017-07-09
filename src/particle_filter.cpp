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

// CURRENT RESULTS:
//
// ERROR
// X: .121
// Y: .114
// Yaw: .004
//
// PERFORMANCE
// System Time: 93.58

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of x, y, theta and their uncertainties
  // from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method
  // (and others in this file).

  num_particles = 100;
  default_random_engine gen;
  Particle particle;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    particles.push_back(particle);
    weights.push_back(1);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  // http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  // http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    double mean_x;
    double mean_y;
    double mean_theta;

    if (yaw_rate == 0) {
      mean_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
      mean_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
      mean_theta = particles[i].theta;
    } else {
      mean_x = particles[i].x + (velocity/yaw_rate) *
               (sin(particles[i].theta + yaw_rate * delta_t) -
               sin(particles[i].theta));
      mean_y = particles[i].y + (velocity/yaw_rate) *
               (-cos(particles[i].theta + yaw_rate * delta_t) +
               cos(particles[i].theta));
      mean_theta = particles[i].theta + yaw_rate * delta_t;
    }

    normal_distribution<double> dist_x(mean_x, std_pos[0]);
    normal_distribution<double> dist_y(mean_y, std_pos[1]);
    normal_distribution<double> dist_theta(mean_theta, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

// void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
//        std::vector<LandmarkObs>& observations)
// TODO: Find the predicted measurement that is closest to each observed
// measurement and assign the observed measurement to this particular landmark.
// NOTE: this method will NOT be called by the grading code. But you will
// probably find it useful to implement this method and use it as a helper
// during the updateWeights phase.

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a multi-variate Gaussian
  // distribution. You can read more about this distribution here:
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located according to the MAP'S coordinate system. You will
  // need to transform between the two systems. Keep in mind that this
  // transformation requires both rotation AND translation (but no scaling).
  // The following is a good resource for the theory:
  // https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  // and the following is a good resource for the actual equation to implement
  // (look at equation 3.33)
  // http://planning.cs.uiuc.edu/node99.html

  // TRANSLATE OBSERVATIONS TO MAP SPACE FOR PARTICLE

  // for weight caclulation
  double normalizer = 1.0f / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  double exponent_x_normalizer = 2 * M_PI * std_landmark[0] * std_landmark[0];
  double exponent_y_normalizer = 2 * M_PI * std_landmark[1] * std_landmark[1];
  double exponent_x;
  double exponent_y;
  double exponent;



  // loop over all particles
  for (int i = 0; i < particles.size(); i++) {
    // declare vector and LandmarkObs to keep track of translations

    LandmarkObs current_transformation;
    std::vector<int> map_associations;
    std::vector<LandmarkObs> transformed_obs;
    long double particle_weight = 1;

    // loop over all observations for each particle
    for (int j = 0; j < observations.size(); j++) {
      // set x, y, and theta values
      double o_x = observations[j].x;
      double o_y = observations[j].y;
      double p_x = particles[i].x;
      double p_y = particles[i].y;
      double p_theta = particles[i].theta;

      // calculate transformation
      current_transformation.x = o_x * cos(p_theta) - o_y * sin(p_theta) + p_x;
      current_transformation.y = o_x * sin(p_theta) + o_y * cos(p_theta) + p_y;

      // add to vector tracking transformations
      transformed_obs.push_back(current_transformation);

      // ASSOCIATE THE OBSERVATIONS WITH THE MAP
      // declare and set variables for distance of closest predicted
      // observation and index
      double min_distance = dist(current_transformation.x,
                                 current_transformation.y,
                                 map_landmarks.landmark_list[0].x_f,
                                 map_landmarks.landmark_list[0].y_f);
      int min_index = 0;

      // loop over remaining predicted observations
      for (int k = 1; k < map_landmarks.landmark_list.size(); k++) {
        // calculate distance of current predicted observation
        double distance = dist(current_transformation.x,
                               current_transformation.y,
                               map_landmarks.landmark_list[k].x_f,
                               map_landmarks.landmark_list[k].y_f);
        // update min distance and index if this is the new min
        if (distance < min_distance) {
          min_distance = distance;
          min_index = k;
        }
      }
      // CALCULATE WEIGHT
      // set values
      // new x_i (or y_i) is location of observation in map space now
      // mu_x is (or mu_y) is location of closest landmark in map space
      double x_i = current_transformation.x;
      double mu_x = map_landmarks.landmark_list[min_index].x_f;
      double y_i = current_transformation.y;
      double mu_y = map_landmarks.landmark_list[min_index].y_f;

      // calculate exponent
      exponent_x = (x_i - mu_x) * (x_i - mu_x)/ (exponent_x_normalizer);
      exponent_y = (y_i - mu_y) * (y_i - mu_y)/ (exponent_y_normalizer);
      exponent = -(exponent_x + exponent_y);
      particle_weight *= normalizer * exp(exponent);
      map_associations.push_back(min_index);
    }
    // Add index to the map associations
    particles[i].associations = map_associations;
    particles[i].weight = particle_weight;
    weights[i] = particle_weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional
  // to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());

  vector<Particle> resample_particles;

  for (int i = 0; i < num_particles; i++) {
    resample_particles.push_back(particles[distribution(gen)]);
  }

  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and
  // association's (x,y) world coordinates mapping to associations: The
  // landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
