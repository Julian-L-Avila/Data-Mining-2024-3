#include <fstream>
#include <random>
#include <string>
#include <cmath>


double RandomDouble(double min, double max) {
	static thread_local std::mt19937 gen(std::random_device{}());
	return std::uniform_real_distribution<double>(min, max)(gen);
}

void GenerateAndSaveData(double ampMin, double ampMax, double freqMin, double freqMax, const std::string& filename) {
	std::ofstream file(filename);
	for (int i = 0; i < 200; ++i) {
		double mass = RandomDouble(0.1, 10.0);
		double amplitude = RandomDouble(ampMin, ampMax);
		double frequency = RandomDouble(freqMin, freqMax);
		double time = RandomDouble(0.0, 10.0);
		double position = amplitude * cos(2.0 * M_PI * frequency * time);
		file << mass << '\t' << amplitude << '\t' << frequency << '\t' << time << '\t' << position << '\n';
	}
}

int main() {
	GenerateAndSaveData(0.5, 10.0, 0.1, 10.0, "train.tsv");
}

