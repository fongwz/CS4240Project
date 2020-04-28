package com.example.cs4240;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;

public class MainActivity extends AppCompatActivity {

    private static final int test_img = R.drawable.i;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ImageView imView = (ImageView)findViewById(R.id.im_view);
        imView.setImageResource(R.drawable.logo);

        ImageView bgIm = findViewById(R.id.background_im);
        bgIm.setBackgroundColor(Color.BLACK);
    }

    public void onButtonClick(View view) {
        Intent i = new Intent(this, Camera2Activity.class);
        startActivity(i);
    }
}
