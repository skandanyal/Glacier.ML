//
// Created by skandan-c-y on 7/25/25.
//

#ifndef MIN_HEAP_HPP
#define MIN_HEAP_HPP

#pragma once
#include <vector>
#include <utility>
#include <iostream>

class MinHeap {
private:
    std::vector<std::pair<double, float>> data;

    void heapify_up(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (data[index].first < data[parent].first) {
                std::swap(data[index], data[parent]);
                index = parent;
            } else {
                break;
            }
        }
    }

    void heapify_down(int index) {
        int size = data.size();

        while (true) {
            int left = 2 * index + 1;
            int right = 2 * index + 2;
            int smallest = index;

            if (left < size && data[left].first < data[smallest].first)
                smallest = left;
            if (right < size && data[right].first < data[smallest].first)
                smallest = right;

            if (smallest != index) {
                std::swap(data[index], data[smallest]);
                index = smallest;
            } else {
                break;
            }
        }
    }

public:
    void push(std::pair<double, float> &value) {
        data.push_back(value);
        heapify_up(data.size() - 1);
    }

    void pop() {
        if (data.empty()) return;
        data[0] = data.back();
        data.pop_back();
        heapify_down(0);
    }

    float top() const {
        if (data.empty()) std::cerr << "Empty Heap\n";
        return data[0].second;
    }

    bool empty() const {
        return data.empty();
    }

    int size() const {
        return data.size();
    }
};

#endif //MIN_HEAP_HPP
