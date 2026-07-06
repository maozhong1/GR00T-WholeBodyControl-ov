#pragma once

template<typename T>
void OVInferenceEngine::SetInputData(const std::string& name, const T* data, size_t elementCount) {
    SetInputData(name, static_cast<const void*>(data), sizeof(T) * elementCount);
}

template<typename T>
void OVInferenceEngine::SetInputData(const std::string& name, const std::vector<T>& data) {
    SetInputData(name, static_cast<const void*>(data.data()), sizeof(T) * data.size());
}

template<typename T>
void OVInferenceEngine::GetOutputData(const std::string& name, T* data, size_t elementCount) {
    GetOutputData(name, static_cast<void*>(data), sizeof(T) * elementCount);
}

template<typename T>
void OVInferenceEngine::GetOutputData(const std::string& name, std::vector<T>& data) {
    GetOutputData(name, static_cast<void*>(data.data()), sizeof(T) * data.size());
}

template<typename T>
void OVInferenceEngine::SetInputDataAsync(const std::string& name, const T* data, size_t elementCount, cudaStream_t) {
    SetInputData(name, data, elementCount);
}

template<typename T>
void OVInferenceEngine::SetInputDataAsync(const std::string& name, const std::vector<T>& data, cudaStream_t) {
    SetInputData(name, data);
}

template<typename T>
void OVInferenceEngine::GetOutputDataAsync(const std::string& name, T* data, size_t elementCount, cudaStream_t) {
    GetOutputData(name, data, elementCount);
}

template<typename T>
void OVInferenceEngine::GetOutputDataAsync(const std::string& name, std::vector<T>& data, cudaStream_t) {
    GetOutputData(name, data);
}
