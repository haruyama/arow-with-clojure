(ns arow-with-clojure.core)

(def DEFAULT_REGULARIZATION 1.0)
(def DEFAULT_MEAN           0.0)
(def DEFAULT_COV            1.0)

(defrecord Model [r mean cov])

(defn initialize-model [& {:keys [r] :or {r DEFAULT_REGULARIZATION}}]
  (if (<= r 0)
    (throw (IllegalArgumentException. "r must be positive")))
  (Model. r {} {}))

(defn- get-mean [mean feature]
  (get mean feature DEFAULT_MEAN))

(defn- get-cov  [cov feature]
  (get cov feature DEFAULT_COV))

(defn- margin [model data]
  (let [mean (:mean model)]
    (reduce-kv (fn [margin f w] (+ margin (* w (get-mean mean f))))
               0.0 data)))

(defn predict [model data]
  (if (>= (margin model data) 0)
    1.0
    -1.0))

(defn- confidence [model data]
  (let [cov (:cov model)]
    (reduce-kv (fn [confidence f w] (+ confidence (* w w (get-cov cov f))))
               0.0 data)))

(defn- update-mean [model label data alpha]
  (let [cov (:cov model)]
    (reduce-kv (fn [m f w]
                 (assoc m f (+ (get-mean m f) (* label alpha (get-cov cov f) w))))
               (:mean model) data)))

(defn- update-cov [model data beta]
  (let [r (:r model)]
    (reduce-kv (fn [c f w]
                 (let [cov_f (get c f 1.0)]
                   (assoc c f (- cov_f
                                 (* (* cov_f cov_f w w) beta)))))
               (:cov model) data)))

(defn update [model label data]
  (let [margin (margin model data)]
    (if (<  (* label margin) 1.0)
      (let [confidence (confidence model data)
            beta       (/ 1.0 (+ confidence (:r model)))
            alpha      (*  (max 0.0 (- 1.0 margin)) beta)]
        (Model. (:r model)
                (update-mean model label data alpha)
                (update-cov  model data beta)))
      model)))
