#pragma once

#include <iostream>
#include "tensor.hpp"
#include "layers.hpp"
#include "helper.hpp"

namespace mygrad {

class Model {
public:

    template<typename... layerTypes>
    Model(layerTypes&&... layers) : 
        layers(layers...),
        parameters(parametersOfLayers(this->layers)),
        nonParameters(nonParametersOfLayers(this->layers)) {}


    void save(const std::string& filename) const;
    void load(const std::string& filename);
    void zeroGrad();
    Tensor& operator()(Tensor& x);
    Tensor& forward(Tensor& x);
    void backward();

    void print() const;
    void printGrads() const;

private:

    class LayersContainer {
    public:

        template <typename... layerTypes>
        LayersContainer(layerTypes&&... layers) {
            this->layers.reserve(sizeof...(layerTypes));

            (add(std::move(layers)), ...);
        }

        inline Layer& operator[](size_t i) { return *layers[i]; }

        inline size_t size() { return layers.size(); }

    private:
        std::vector<std::unique_ptr<Layer>> layers;

        template <typename layerType>
        void add(layerType&& layer) {
            layers.push_back(std::make_unique<layerType>(std::move(layer)));
        }
    };

    LayersContainer layers;

    const std::vector<Tensor*> parametersOfLayers(LayersContainer& layers);
    const std::vector<Tensor*> nonParametersOfLayers(LayersContainer& layers);

public:
    const std::vector<Tensor*> parameters;
    const std::vector<Tensor*> nonParameters;
};

} // namespace mygrad