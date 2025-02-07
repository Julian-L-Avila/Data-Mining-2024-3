#include <fstream>
#include <random>
#include <cmath>

double RandomDouble(double min, double max) {
	static thread_local std::mt19937 gen(std::random_device{}());
	return std::uniform_real_distribution<double>(min, max)(gen);
}

double RadioactiveDecay(double initial, double halflife, double time) {
	return initial * std::pow(0.5, time / halflife);
}

void GenerateRandomDecayData(const std::string& filename) {
	std::ofstream file(filename);
	for (int i = 0; i < 200; ++i) {
		double initial = RandomDouble(10.0, 100.0);
		double halflife = RandomDouble(1.0, 10.0);
		double time = RandomDouble(0.0, 50.0);
		double remaining = RadioactiveDecay(initial, halflife, time);
		file << initial << '\t' << halflife << '\t' << time << '\t' << remaining << '\n';
	}
}

void GenerateFixedDecayData(double initial, double halflife, const std::string& filename) {
	std::ofstream file(filename);
	for (int i = 0; i < 200; ++i) {
		double time = i * 0.5;
		double remaining = RadioactiveDecay(initial, halflife, time);
		file << initial << '\t' << halflife << '\t' << time << '\t' << remaining << '\n';
	}
}

int main() {
	GenerateRandomDecayData("random-decay.tsv");
	GenerateFixedDecayData(50.0, 5.0, "fixed-decay.tsv");
}

