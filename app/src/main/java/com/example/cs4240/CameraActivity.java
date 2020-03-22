package com.example.cs4240;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.MenuItem;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.firebase.ml.common.FirebaseMLException;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CameraActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    /*
    private CameraManager cameraManager;
    int cameraFacing;
    private TextureView.SurfaceTextureListener surfaceTextureListener;
    private String cameraId;

    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private CameraDevice.StateCallback stateCallback;
    private CameraCaptureSession cameraCaptureSession;
    private CameraDevice cameraDevice;
    private Size previewSize;
    private CaptureRequest.Builder captureRequestBuilder;
    private CaptureRequest captureRequest;
    private ImageReader imageReader;

    private TextureView textureView;

    private static final int CAMERA_REQUEST_CODE = 1888;

    private HandClassifier classifier;
    private Bitmap newBmp;
    */

    private static final String TAG = "test";

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean  mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;

    Mat mRgba;
    Mat mRgbaF;
    Mat mRgbaT;
    private Bitmap bmp;
    private Bitmap overlayBmp;
    private boolean initBmp = false;
    private static final int CAMERA_PERMISSION = 555;
    private HandClassifier classifier;

    private CameraCharacteristics cameraCharacteristics;
    private CameraManager manager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_camera);

        //previewCameraSizes();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
        }

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.cv_camera_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMinimumWidth(720); //doesn't seem to work
        mOpenCvCameraView.setMinimumHeight(1280);
        mOpenCvCameraView.setMaxFrameSize(720,1280);

        DisplayMetrics displayMetrics = getResources().getDisplayMetrics();
        float dpHeight = displayMetrics.heightPixels / displayMetrics.density;
        float dpWidth = displayMetrics.widthPixels / displayMetrics.density;
        float height = (dpWidth *1280)/720;
        overlayBmp = Bitmap.createBitmap(720,1280, Bitmap.Config.ARGB_8888);
        overlayBmp=Bitmap.createScaledBitmap(overlayBmp, 720 , (int)height, true);

        ImageView imView = (ImageView)findViewById(R.id.im_view);
        imView.setImageBitmap(overlayBmp);
        try {
            classifier = new HandClassifier(this, "palm.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        if (!initBmp) {
            bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
            initBmp = true;
        }
        // Rotate mRgba 90 degrees
        Core.transpose(mRgba, mRgbaT);
        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 720,1280, 0);
        Core.flip(mRgbaF, mRgba, 1 );

        Utils.matToBitmap(mRgba, bmp);
        overlayBmp.eraseColor(Color.TRANSPARENT);

        try {
            classifier.predict(bmp);
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
        return mRgba; // This function must return
    }

    public void setImage() {
        this.classifier.label(overlayBmp);
        ImageView imView = (ImageView)findViewById(R.id.im_view);
        imView.setImageBitmap(overlayBmp);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,  String permissions[], int[] grantResults) {
        switch (requestCode) {
            case CAMERA_PERMISSION:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    //if(mClss != null) {
                        Intent intent = new Intent(this, CameraActivity.class);
                        startActivity(intent);
                    //}
                } else {
                    Toast.makeText(this, "Please grant camera permission to use the QR Scanner", Toast.LENGTH_SHORT).show();
                }
                return;
        }
    }

    private void previewCameraSizes() {
        // Check camera supported sizes
        manager = (CameraManager) this.getSystemService(Context.CAMERA_SERVICE);
        try {
            cameraCharacteristics = manager.getCameraCharacteristics(manager.getCameraIdList()[0]);
            StreamConfigurationMap scmap = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size[] previewSizes = scmap.getOutputSizes(ImageReader.class);
            for (int i = 0; i < previewSizes.length; i++) {
                //Log.d("test", previewSizes[i].toString());
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
}
