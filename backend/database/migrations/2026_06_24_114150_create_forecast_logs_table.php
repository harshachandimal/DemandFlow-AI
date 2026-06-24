<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('forecast_logs', function (Blueprint $table) {
            $table->id();
            $table->integer('store_id');
            $table->json('request_payload');
            $table->json('response_payload');
            $table->integer('total_predicted_sales')->nullable();
            $table->integer('average_predicted_sales')->nullable();
            $table->string('stockout_risk')->nullable();
            $table->boolean('reorder_needed')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('forecast_logs');
    }
};
