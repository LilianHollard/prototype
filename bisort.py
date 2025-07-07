import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou_1d_weighted(bb_test, bb_gt, x_weight=0.7):
    """IoU optimisé pour mouvement 1D avec pondération sur l'axe X"""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    
    # Calcul IoU standard
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - intersection
    
    if union <= 0:
        return 0.0
    
    iou_standard = intersection / union
    
    # Bonus pour alignement horizontal (axe X prioritaire)
    x_overlap = w / max(bb_test[2] - bb_test[0], bb_gt[2] - bb_gt[0])
    
    return iou_standard * (1 + x_weight * x_overlap)

def distance_1d(bb_test, bb_gt):
    """Distance 1D simplifiée pour mouvement unidirectionnel"""
    center_test_x = (bb_test[0] + bb_test[2]) / 2
    center_gt_x = (bb_gt[0] + bb_gt[2]) / 2
    return abs(center_test_x - center_gt_x)

#TODO : estimate velocity with GPS !!
def estimate_velocity_from_detections(current_bbox, previous_bbox):
    """Estime la vitesse en X basée sur les détections précédentes"""
    if previous_bbox is None:
        return 0.0
    
    current_center_x = (current_bbox[0] + current_bbox[2]) / 2
    previous_center_x = (previous_bbox[0] + previous_bbox[2]) / 2
    
    return current_center_x - previous_center_x

class KalmanBoxTracker1D:
    """Tracker Kalman optimisé pour mouvement 1D"""
    count = 0
    
    def __init__(self, bbox, initial_velocity=None, camera_speed_px_per_frame=3.5):
        # Modèle de vitesse constante simplifié pour 1D
        # États: [x, y, s, r, vx, vy, vs] - focus sur vx
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # Matrice de transition optimisée pour mouvement 1D
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0],  # y = y + vy (moins important)
            [0, 0, 1, 0, 0, 0, 1],  # s = s + vs
            [0, 0, 0, 1, 0, 0, 0],  # r = r (ratio constant)
            [0, 0, 0, 0, 1, 0, 0],  # vx = vx (vitesse X constante)
            [0, 0, 0, 0, 0, 0.8, 0], # vy décroît (mouvement principalement 1D)
            [0, 0, 0, 0, 0, 0, 0.9]  # vs décroît lentement
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Matrices de covariance optimisées
        self.kf.R[2:, 2:] *= 10.  # Incertitude taille/ratio
        self.kf.P[4:, 4:] *= 1000.  # Incertitude vitesses initiales
        self.kf.P *= 10.
        
        # Bruit de processus adapté au mouvement 1D
        self.kf.Q[-1, -1] *= 0.01  # Faible bruit sur vs
        self.kf.Q[4, 4] *= 0.01    # Très faible bruit sur vx (vitesse constante)
        self.kf.Q[5, 5] *= 0.05     # Bruit moyen sur vy
        self.kf.Q[6, 6] *= 0.01     # Faible bruit sur vs
        
        # Initialisation avec vitesse estimée de la caméra
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        
        #self.kf.x[4] = camera_speed_px_per_frame  # Vitesse positive (objets vont vers la droite)
        #objets vont vers la droite ou vers la gauche (changement de position avec le rang de vigne)
        if initial_velocity is not None:
            self.kf.x[4]=initial_velocity
        else:
            self.kx.x[4] = 0.0
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker1D.count
        KalmanBoxTracker1D.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.camera_speed = camera_speed_px_per_frame
        
        #estimation pour la vitesse
        self.previous_bbox = bbox
        self.velocity_history = []
        self.velocity_estimation_frames = 3

    def update(self, bbox):
        """Mise à jour avec nouvelle détection"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        if self.previous_bbox is not None:
            estimated_velocity = estimate_velocity_from_detections(bbox, self.previous_bbox)
            self.velocity_history.append(estimated_velocity)

            if len(self.velocity_history) > self.velocity_estimation_frames:
                self.velocity_history.pop(0)
            
            if len(self.velocity_history) >= 2:
                avg_velocity = np.mean(self.velocity_history)

                #MAJ douce de la vitesse prédite
                self.kf.x[4] = 0.7 * self.kf.x[4] + 0.3 * avg_velocity

        self.kf.update(self.convert_bbox_to_z(bbox))
        self.previous_bbox = bbox

    def predict(self):
        """Prédiction optimisée pour 1D"""
        # Évite les tailles négatives
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.time_since_update += 1
        
        # Garde seulement la prédiction actuelle (pas d'historique inutile)
        bbox = self.convert_x_to_bbox(self.kf.x)
        return bbox

    def get_state(self):
        """État actuel du tracker"""
        return self.convert_x_to_bbox(self.kf.x)

    def is_out_of_bounds(self, frame_width):
        """Vérifie si l'objet est sorti du cadre"""
        bbox = self.get_state()[0]
        return bbox[2] < 0 or bbox[0] > frame_width 

    def get_velocity(self):
        return self.kf.x[4]

    @staticmethod
    def convert_bbox_to_z(bbox):
        """Conversion bbox vers état Kalman"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([[x], [y], [s], [r]])

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """Conversion état Kalman vers bbox"""
        if x[2] <= 0:  # Protection contre les aires négatives
            w, h = 1, 1
        else:
            w = np.sqrt(x[2] * x[3])
            h = x[2] / w if w > 0 else 1
            
        bbox = np.array([
            x[0] - w/2., 
            x[1] - h/2., 
            x[0] + w/2., 
            x[1] + h/2.
        ]).reshape((1, 4))
        
        if score is not None:
            bbox = np.concatenate((bbox, np.array([[score]])), axis=1)
        return bbox

class Sort1D:
    """SORT optimisé pour tracking 1D de grappes de raisin"""
    
    def __init__(self, 
                 camera_speed_kmh=3.75,  # Vitesse moyenne 3.5-4 km/h
                 fps=30,
                 frame_width=640,       # Largeur de l'image
                 min_hits=2,             # Réduit car mouvement prévisible
                 iou_threshold=0.25,
                 velocity_estimation_frames=3):    # Seuil plus bas car 1D
        
        # Calcul des paramètres optimaux
        self.fps = fps
        self.frame_width = frame_width
        
        # Conversion vitesse en pixels par frame
        # Approximation: 1 m ≈ 100 pixels (à ajuster selon votre setup)
        # Approximation: 1 m = taille d'image
        speed_ms = camera_speed_kmh * 1000 / 3600  # m/s
        self.camera_speed_px_per_frame = speed_ms * (frame_width / fps)  # px/frame
        
        # Calcul du max_age optimal
        # Temps pour qu'une grappe traverse complètement l'écran
        frames_to_cross_screen = frame_width / self.camera_speed_px_per_frame
        self.max_age = max(1, int(frames_to_cross_screen * 0.3))  # 30% du temps de traversée
        #self.max_age = 5
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

        self.velocity_estimation_frames = velocity_estimation_frames
        self.detection_history = {}
        
        print(f"SORT 1D initialisé:")
        print(f"  - Vitesse caméra: {self.camera_speed_px_per_frame:.1f} px/frame")
        print(f"  - Max age: {self.max_age} frames")
        print(f"  - Min hits: {self.min_hits}")

    def estimate_initial_velocity(self, bbox, detection_id=None):
        if detection_id is None:
            return 0.0

        if detection_id in self.detection_history:
            previous_bbox = self.detection_history[detection_id]
            velocity = estimate_velocity_from_detections(bbox, previous_bbox)
            return velocity

        return 0.0

    def update(self, dets=np.empty((0, 5))):
        """Mise à jour principale du tracker"""
        self.frame_count += 1
        
        # Prédiction pour tous les trackers existants
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, tracker in enumerate(self.trackers):
            pos = tracker.predict()[0]
            trks[t, :4] = pos
            trks[t, 4] = 0
            
            # Supprime les trackers avec prédictions invalides
            if np.any(np.isnan(pos)) or tracker.is_out_of_bounds(self.frame_width):
                to_del.append(t)
        
        # Nettoyage des trackers invalides AVANT l'association
        valid_trackers = []
        valid_trks = []
        
        for t, tracker in enumerate(self.trackers):
            if t not in to_del:
                valid_trackers.append(tracker)
                valid_trks.append(trks[t])
        
        # Mise à jour de la liste des trackers
        self.trackers = valid_trackers
        
        if len(valid_trks) > 0:
            trks = np.array(valid_trks)
        else:
            trks = np.empty((0, 5))

        # Association détections-trackers optimisée 1D
        if len(dets) > 0 and len(trks) > 0:
            matched, unmatched_dets, unmatched_trks = self.associate_detections_1d(dets, trks)
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(len(dets))
            unmatched_trks = np.arange(len(trks))

        # Mise à jour des trackers matchés
        for m in matched:
            if m[1] < len(self.trackers):  # Protection contre l'indexation
                self.trackers[m[1]].update(dets[m[0], :4])

        # Création de nouveaux trackers
        for i in unmatched_dets:
            if i < len(dets):  # Protection contre l'indexation
                bbox = dets[i, :4]
                initial_velocity = self.estimate_initial_velocity(bbox,i)
                trk = KalmanBoxTracker1D(bbox, initial_velocity, self.camera_speed_px_per_frame)
                self.trackers.append(trk)

        # Génération des résultats et nettoyage
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            # Conditions de validation plus strictes
            if ((trk.time_since_update < 1) and 
                (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            
            i -= 1
            
            # Suppression des trackers expirés
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(ret) if len(ret) > 0 else np.empty((0, 5))

    def associate_detections_1d(self, detections, trackers):
        """Association optimisée pour mouvement 1D"""
        if len(trackers) == 0:
            return (np.empty((0, 2), dtype=int), 
                   np.arange(len(detections)), 
                   np.empty((0), dtype=int))
        
        # Matrice de coût hybride (IoU + distance 1D)
        cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                # IoU pondéré pour 1D
                iou_score = iou_1d_weighted(det, trk, x_weight=0.8)
                
                # Distance 1D normalisée
                dist_1d = distance_1d(det, trk) / self.frame_width
                
                # Bonus pour cohérence de direction (si on a l'historique)
                direction_bonus = 1.0
                if hasattr(self.trackers[t], 'get_velocity'):
                    predicted_velocity = self.trackers[t].get_velocity()
                    # Petit bonus si la direction est cohérente
                    if abs(predicted_velocity) > 0.5:  # Seuil minimal de vitesse
                        det_center_x = (det[0] + det[2]) / 2
                        trk_center_x = (trk[0] + trk[2]) / 2
                        actual_movement = det_center_x - trk_center_x
                        
                        # Bonus si le mouvement prédit et réel sont dans la même direction
                        if (predicted_velocity > 0 and actual_movement > 0) or \
                           (predicted_velocity < 0 and actual_movement < 0):
                            direction_bonus = 1.2
                
                # Score combiné
                cost_matrix[d, t] = iou_score * (1 - dist_1d) * direction_bonus
        
        # Association par algorithme hongrois
        matched_indices = linear_sum_assignment(-cost_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        # Filtrage des associations
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Validation des matches avec seuil adaptatif
        matches = []
        for m in matched_indices:
            if cost_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# Fonction utilitaire pour faciliter l'utilisation
def create_grape_tracker(camera_speed_kmh=3, min_hits=2,fps=60, frame_width=640):
    """
    Crée un tracker optimisé pour grappes de raisin
    
    Args:
        camera_speed_kmh: Vitesse de la caméra en km/h
        fps: Images par seconde
        frame_width: Largeur de l'image en pixels
    """
    return Sort1D(
        camera_speed_kmh=camera_speed_kmh,
        fps=fps,
        min_hits=min_hits,
        frame_width=frame_width
    )