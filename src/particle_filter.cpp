/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <float.h>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  default_random_engine gen;
  
  num_particles = 50;  // Set the number of particles
  
  // This creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) 
  {
    Particle new_particle;
    new_particle.id = i;
    
    // TODO: Sample from these normal distributions like this: 
    new_particle.x = dist_x(gen);
    new_particle.y = dist_y(gen);
    new_particle.theta = dist_theta(gen);
	new_particle.weight = 1;
    
    particles.push_back(new_particle);
    weights.push_back(new_particle.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) 
{
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
 
  // predictin equations:
  // x_f = x_0 + v/theta_dot * (sin(theta_0 + theta_dot*delta_time) - sin(theta_0) )
  // y_f = y_0 + v/theta_dot * (cos(theta_0) - cos(theta_0 + theta_dot*delta_time) )
  // theta_f = theta_0 + theta_dot * dt
  for (int i = 0; i < num_particles; i++)
  {
    double predicted_x;
    double predicted_y;
    double predicted_theta;
    
    if (fabs(yaw_rate) < 0.0001) //if the yaw_rate is almost zero
    {
      //prediction equations with very small yaw_rate
      predicted_x = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
	  predicted_y = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
      predicted_theta = particles[i].theta;
    }
    else
    {
      //normal prediction equations
      predicted_x = particles[i].x + (velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)));
      predicted_y = particles[i].y + (velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)));
      predicted_theta = particles[i].theta + yaw_rate * delta_t;
    }
    
    //add noise
    normal_distribution<double> noise_x(predicted_x, std_pos[0]);
    normal_distribution<double> noise_y(predicted_y, std_pos[1]);
    normal_distribution<double> noise_theta(predicted_theta, std_pos[2]);
    
    particles[i].x = noise_x(gen);
    particles[i].y = noise_y(gen);
    particles[i].theta = noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) 
{
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for(unsigned int i = 0; i < observations.size(); i++)
  {
    //set distance to the maximum value a double can get
    double distance = DBL_MAX; 
    int map_id = -1;
    for(unsigned int j = 0; j < predicted.size(); j++)
    {
      double current_distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      if(current_distance <  distance)
      {
        distance = current_distance;
        map_id = predicted[j].id;
      }
    }
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  /*This variable is used for normalizing weights of all particles and bring them in the range
    of [0, 1]*/
  double weight_normalizer = 0.0;

  for (int i = 0; i < num_particles; i++) 
  {
    /*Step 1: Transform observations from vehicle co-ordinates to map co-ordinates.*/
    //Vector containing observations transformed to map co-ordinates w.r.t. current particle.
    vector<LandmarkObs> transformed_observations;

    //Transform observations from vehicle's co-ordinates to map co-ordinates.
    for (unsigned int j = 0; j < observations.size(); j++) 
    {
      LandmarkObs transformed_obs;
      transformed_obs.id = observations[j].id;
      transformed_obs.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
      transformed_obs.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
      transformed_observations.push_back(transformed_obs);
    }

    /*Step 2: Filter map landmarks to keep only those which are in the sensor_range of current 
     particle. Push them to predictions vector.*/
    vector<LandmarkObs> predicted_landmarks;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) 
    {
      Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
      if ((fabs((particles[i].x - current_landmark.x_f)) <= sensor_range)
          && (fabs((particles[i].y - current_landmark.y_f)) <= sensor_range)) 
      {
        predicted_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    }

    /*Step 3: Associate observations to lpredicted andmarks using nearest neighbor algorithm.*/
    //Associate observations with predicted landmarks
    dataAssociation(predicted_landmarks, transformed_observations);

    /*Step 4: Calculate the weight of each particle using Multivariate Gaussian distribution.*/
    //Reset the weight of particle to 1.0
    particles[i].weight = 1.0;

    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double sigma_x_2 = pow(sigma_x, 2);
    double sigma_y_2 = pow(sigma_y, 2);
    double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
    
    /*Calculate the weight of particle based on the multivariate Gaussian probability function*/
    for (unsigned int k = 0; k < transformed_observations.size(); k++) 
    {
      double multi_prob = 1.0;

      for (unsigned int l = 0; l < predicted_landmarks.size(); l++)
     {
        if (transformed_observations[k].id == predicted_landmarks[l].id) 
        {
          multi_prob = normalizer * exp(-1.0 * ((pow((transformed_observations[k].x - predicted_landmarks[l].x), 2)/(2.0 * sigma_x_2)) 
                                               + (pow((transformed_observations[k].y - predicted_landmarks[l].y), 2)/(2.0 * sigma_y_2))));
          particles[i].weight *= multi_prob;
        }
      }
    }
    weight_normalizer += particles[i].weight;
  }

  /*Step 5: Normalize the weights of all particles since resmapling using probabilistic approach.*/
  for (unsigned int i = 0; i < particles.size(); i++) 
  {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() 
{
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	vector<Particle> new_particles;
    default_random_engine gen;

    uniform_int_distribution<> int_dist(0, num_particles - 1);
    int index = int_dist(gen);

    double max_weight = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> real_dist(0.0, 2*max_weight);

    double beta = 0;
    for (int i = 0; i < num_particles; i++) 
    {
        beta += real_dist(gen);

        while(beta > weights[index]) 
        {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) 
{
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) 
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) 
{
  vector<double> v;

  if (coord == "X") 
  {
    v = best.sense_x;
  } 
  else 
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
