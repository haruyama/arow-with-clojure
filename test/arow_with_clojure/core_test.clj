(ns arow-with-clojure.core-test
  (:require [clojure.test :refer :all]
            [arow-with-clojure.core]))

(def train-data [
  [ 1.0  {"R" 255 "G" 0   "B" 0}]
  [ -1.0 {"R" 0   "G" 255 "B" 0}]
  [ -1.0 {"R" 0   "G" 0   "B" 255}]
  [ -1.0 {"R" 0   "G" 255 "B" 255}]
  [ 1.0  {"R" 255 "G" 0   "B" 255}]
  [ 1.0  {"R" 255 "G" 255 "B" 0}]])


(defn- train [model train-data]
  (reduce (fn [m entry]
               (arow-with-clojure.core/update m (first entry) (second entry)))
             model train-data))

(deftest arow
         (let [model (train (arow-with-clojure.core/initialize-model) train-data)]
           (doseq [entry train-data]
             (is (= true (>  (* (first entry) (arow-with-clojure.core/classify model (second entry))) 0))))))

(deftest arow-2
         (let [model (train (arow-with-clojure.core/initialize-model :r 0.1) train-data)]
           (doseq [entry train-data]
             (is (= true (>  (* (first entry) (arow-with-clojure.core/classify model (second entry))) 0))))))
