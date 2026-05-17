#include "musiq_postprocess.h"

namespace musiq {

float distToMos(const std::vector<float>& dist_score) {
    float mos_score = 0.0f;
    
    for (size_t i = 0; i < dist_score.size(); ++i) {
        mos_score += dist_score[i] * static_cast<float>(i + 1);
    }
    
    return mos_score;
}

} 
