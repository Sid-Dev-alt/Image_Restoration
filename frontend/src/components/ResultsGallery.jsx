import React from 'react';

export default function ResultsGallery({ images, sessionId, onCompare }) {
  if (images.length === 0) {
    return (
      <div className="card border-light-subtle bg-white text-center p-5 shadow-sm text-secondary">
        <i className="bi bi-images fs-1 mb-3 text-muted"></i>
        <p className="mb-0">No images restored yet.</p>
        <small className="text-muted">Upload a folder and trigger restoration to view results.</small>
      </div>
    );
  }

  const getModelBadge = (modelUsed) => {
    let text = 'Restored';
    let className = 'bg-secondary';

    if (modelUsed === 'nafnet_gopro') {
      text = 'NAFNet-GoPro';
      className = 'bg-success';
    } else if (modelUsed === 'restormer_motion') {
      text = 'Restormer-Motion';
      className = 'bg-purple'; // custom color logic can be added or standard bootstrap values
    } else if (modelUsed === 'restormer_defocus') {
      text = 'Restormer-Defocus';
      className = 'bg-primary';
    } else if (modelUsed === 'mprnet_deblur') {
      text = 'MPRNet-Deblur';
      className = 'bg-warning text-dark';
    }

    return { text, className };
  };

  return (
    <div>
      <h5 className="fw-bold mb-3 text-dark">Restoration Results</h5>
      <div className="row g-3">
        {images.map((img, idx) => {
          const restoredUrl = `/api/image/restored/${sessionId}/${img.filename}`;
          const badgeInfo = getModelBadge(img.model_used);
          return (
            <div key={idx} className="col-12 col-md-6 col-lg-4">
              <div className="card h-100 border-light-subtle bg-white shadow-sm overflow-hidden result-card">
                <div
                  className="card-media bg-dark position-relative"
                  style={{
                    paddingTop: '66.6%',
                    overflow: 'hidden'
                  }}
                >
                  <img
                    src={restoredUrl}
                    alt={img.filename}
                    className="position-absolute w-100 h-100 top-0 start-0"
                    style={{ objectFit: 'cover' }}
                  />
                </div>
                <div className="card-body p-3 d-flex flex-column justify-content-between">
                  <div
                    className="fw-bold text-dark text-truncate mb-3"
                    title={img.filename}
                    style={{ fontSize: '0.92rem' }}
                  >
                    {img.filename}
                  </div>
                  <div className="d-flex justify-content-between align-items-center">
                    <span className={`badge ${badgeInfo.className} py-1 px-2`}>
                      {badgeInfo.text}
                    </span>
                    <button
                      onClick={() => onCompare(img.filename)}
                      className="btn btn-outline-secondary btn-sm"
                    >
                      Compare
                    </button>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
