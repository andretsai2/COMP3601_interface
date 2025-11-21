/******************************************************************************
* Copyright (C) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xil_io.h"
#include "sleep.h"
#include "xaxidma.h"
#include "xilffs.h"
#include "ff.h"
//#include "time.h"
#include "xtime_l.h"

#define AXI_GPIO_LED_OFFSET 0
#define AXI_GPIO_SW_OFFSET 4
#define N 1
#define BUF_SIZE 1000

volatile u32 audio_rx_buf[BUF_SIZE] = {0};

// for sd card transfer

FATFS FatFs;

// for state machine

enum board_state {
	recording_off,
	recording_on
};

enum button_state {
	released,
	pressed
};

struct fsm {
	enum button_state curr_button_state;
	enum button_state prev_button_state;
};

// for audio capture
void audio_capture();
static void write_le_u32(uint8_t *p, uint32_t v);
static void write_le_u16(uint8_t *p, uint16_t v);
static void fill_wav_header(uint8_t *hdr, uint32_t sample_rate, uint16_t bits_per_sample, uint16_t channels, uint32_t data_bytes);

// for logging file
static void log_time(int command);
static int get_seconds();
static void format_hms(int seconds, char *buf);
static void wipe_log_file();

// global variables for log
static int recordings_counter = 1;
static double cur_time_starts = 0;
static double cur_time_ends = 0;

int main()
{
    init_platform();
    print("Startingup\n\r");

    // fsm initiate
    enum board_state current_board_state = recording_off;
    struct fsm curfsm;

    // initiate values
    int val_read = Xil_In32(XPAR_AXI_GPIO_0_BASEADDR + AXI_GPIO_SW_OFFSET);
	int low_3_bits = (val_read & 0b111); // this indicates button activity
	curfsm.curr_button_state = low_3_bits;
	curfsm.prev_button_state = curfsm.curr_button_state;

	wipe_log_file();

	while (1) {
		val_read = Xil_In32(XPAR_AXI_GPIO_0_BASEADDR + AXI_GPIO_SW_OFFSET);
		low_3_bits = (val_read & 0b111);

		curfsm.prev_button_state = curfsm.curr_button_state;
		curfsm.curr_button_state = low_3_bits;

		// if state goes from pressed to released, it means it's time to switch states
		// debounce logic can possibly go here
		if ((curfsm.prev_button_state == pressed) && (curfsm.curr_button_state == released)) {
			if (current_board_state == recording_off) {
				current_board_state = recording_on;
			} else {
				current_board_state = recording_off;
			}
		}
		// use this code if there are more than 2 states
//		if ((curfsm.prev_button_state == pressed) && (curfsm.curr_button_state == released)) {
//			current_board_state = current_board_state + 1;
//		}

//		// and if the board state exceeds enum we reset it back to 0
//		if (current_board_state >= 2) {
//			current_board_state = recording_off;
//		}

		// all possible board states
		switch (current_board_state) {
			case recording_off: // led is continuously off
				Xil_Out32(XPAR_AXI_GPIO_0_BASEADDR + AXI_GPIO_LED_OFFSET, 0);
				break;

			case recording_on: // led is continuously on and audio records
				Xil_Out32(XPAR_AXI_GPIO_0_BASEADDR + AXI_GPIO_LED_OFFSET, 1);
				audio_capture();
				current_board_state = recording_off;
				break;
		}

	}

    cleanup_platform();
    return 0;
}

void audio_capture() {

	// initialising the dma

    int Status = XST_SUCCESS;
    XAxiDma_Config *CfgPtr;
    XAxiDma AxiDma;

    CfgPtr = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
    if (!CfgPtr) {
        print("No CfgPtr\r\n");
        return;
    }

    Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
    if (Status != XST_SUCCESS) {
        print("DMA cfg init failure\r\n");
        return;
    }

    if (XAxiDma_HasSg(&AxiDma)) {
        print("Device configured as SG mode\r\n");
        return;
    }

    print("DMA initialised\r\n");

    Status = XAxiDma_Selftest(&AxiDma);
    if (Status != XST_SUCCESS) {
        print("DMA failed selftest\r\n");
        return ;
    }

    print("DMA passed self test\r\n");

    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    // do file transfer

    // sc card's FatFs variables
    FRESULT fres;
    FIL wavfile;
    UINT bw;
    uint32_t total_data_bytes = 0;

    // mount sd card
    fres = f_mount(&FatFs, "", 1);
    if (fres != FR_OK) {
        xil_printf("f_mount failed: %d\r\n", fres);
        return;
    } else {
        xil_printf("SD card mounted\r\n");
    }

    // make a wav header (the wav header is the first 44 bytes of the .wav file that contains file info, like file format, sample rate, number of bits blahblahblah)
    uint8_t wav_header[44];
    uint32_t sample_rate = 16000;
    uint16_t bits_per_sample = 16; // originally 32?
    uint16_t channels = 1; // originally 2
    fill_wav_header(wav_header, sample_rate, bits_per_sample, channels, 0);

    // create/overwrite wav on sd card --> 0:/recording.wav
    char wav_file_name[255]; // 255 from FF_MAX_LFN from ff.h from ffconf.h
    snprintf(wav_file_name, sizeof(wav_file_name), "REC%d.WAV", recordings_counter);
    fres = f_open(&wavfile, wav_file_name, FA_CREATE_ALWAYS | FA_WRITE);
    if (fres != FR_OK) {
        xil_printf("f_open failed: %d\r\n", fres);
        return;
    }

    // placeholder header for now
    fres = f_write(&wavfile, wav_header, 44, &bw);
    if (fres != FR_OK || bw != 44) {
        xil_printf("f_write header failed: %d bw=%d\r\n", fres, bw);
        f_close(&wavfile);
        return;
    }

    // for conversion buffer convert u32 -> int16_t
    int16_t pcm_buf[BUF_SIZE];
    // int16_t pcm_16k_buf[BUF_SIZE/2];

    // flush cache
    Xil_DCacheFlushRange((UINTPTR)audio_rx_buf, BUF_SIZE * sizeof(u32));

//    print("starting audio capture\r\n");
    xil_printf("starting audio capture %d\r\n", recordings_counter);

    // log recording start time
    log_time(0);

    // capture audio in loop using the dma
    int counter = 0;
    int stop_recording = 0;
    int val = Xil_In32(XPAR_AXI_GPIO_0_BASEADDR + AXI_GPIO_SW_OFFSET);
	int sw = val & 0b111;
	int prev_sw;
    while (counter < 3000 && !stop_recording) {
    	// first check state change
    	prev_sw = sw;
    	val = Xil_In32(XPAR_AXI_GPIO_0_BASEADDR + AXI_GPIO_SW_OFFSET);
    	sw = val & 0b111;
    	// detect falling edge (pressed to released)
		if (prev_sw == pressed && sw == released) {
			xil_printf("Recording stopped by state change (recording_off)\r\n");
			stop_recording = 1;
			break;
		}
		prev_sw = sw;

        counter++;

        /* We will need to always flush the buffers before
        * using DMA-managed memory,
        * unless we properly configure cache coherency */
        Xil_DCacheFlushRange((UINTPTR)audio_rx_buf, BUF_SIZE * sizeof(u32));

        // start dma transfer here
        Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)audio_rx_buf, BUF_SIZE * sizeof(u32), XAXIDMA_DEVICE_TO_DMA);
        if (Status != XST_SUCCESS) {
            xil_printf("failed rx transfer call\r\n");
            break;
        }

//        print("rx calls good\r\n");

		// At this point in execution, our DMA would now be running and transferring everything from the tx_buf to the rx_buf!
		// Wait until execution completed
        while (XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) {
            usleep(1);
        }

        Xil_DCacheInvalidateRange((UINTPTR)audio_rx_buf, BUF_SIZE * sizeof(u32));

        /*
         * From analysis earlier, we can see that it always behave like like:
         * buf[0] = data;
         * buf[1] = 0;
         * buf[2] = data;
         * buf[3] = 0;
         *
         * So we simply sample every 2 index. Since the original sampling rate is 32KHz.
         * by halving the sampling, we actually get 16KHz, unintentionally. The audio also
         * seems fine without low pass filter, since it's every second one is empty,
         * we are not deleting audio and causing distortion.
         *
         * However, this follows several restrictions:
         * 1. We must run mono-audio.
         * 2. The sampling rate has to be 32KHz, otherwise, we'll have to recalculated
         * 		and redo the loop below.
         */

        int sum_even = 0, sum_odd = 0;
        int check_len = 64; // how many words to sample for detection (<= BUF_SIZE)
        if (check_len > BUF_SIZE) check_len = BUF_SIZE;

        // determine whether audio is in every odd or even buffer
        for (int i = 0; i < check_len; ++i) {
            int16_t v = (int16_t)(audio_rx_buf[i] >> 16); // upper half
            if ((i & 1) == 0) sum_even += abs((int)v);
            else sum_odd  += abs((int)v);
        }
        int start_index = (sum_odd > sum_even) ? 1 : 0;

        // put chosen odd/even buffers into pcm_buf
        int pcm_i = 0;
        for (int i = start_index; i < BUF_SIZE; i += 2) {
            int32_t s = ((int32_t)audio_rx_buf[i]) >> 16; // arithmetic shift to preserve sign
            pcm_buf[pcm_i++] = (int16_t)s;
        }
        int pcm_samples = pcm_i;

        // write pcm to wav
        fres = f_write(&wavfile, pcm_buf, pcm_samples * sizeof(int16_t), &bw);
        if (fres != FR_OK) {
            xil_printf("f_write data failed: %d\n", fres);
        } else if (bw != (UINT)(pcm_samples * sizeof(int16_t))) {
            xil_printf("f_write wrote %u bytes (expected %u)\n",
                       bw, (unsigned)(pcm_samples * sizeof(int16_t)));
        }
        total_data_bytes += bw;

        if ((counter % 100) == 0) {
            xil_printf("Wrote %d chunks. Total bytes: %u\r\n", counter, total_data_bytes);
            f_sync(&wavfile); //  this for header update
        }
    }

    // log recording end time
    log_time(1);

    // update wav header with actual size
    fill_wav_header(wav_header, sample_rate, bits_per_sample, channels, total_data_bytes);

    // seek to start and write corrected header
    f_lseek(&wavfile, 0);
    fres = f_write(&wavfile, wav_header, 44, &bw);
    if (fres != FR_OK || bw != 44) {
        xil_printf("f_write header patch failed: %d bw=%d\r\n", fres, bw);
    } else {
        xil_printf("WAV header patched, final data bytes = %u\r\n", total_data_bytes);
    }

    f_close(&wavfile);
    print("Finished writing WAV to SD\r\n");
}

// helper: write little-endian 32-bit and 16-bit
static void write_le_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)((v >> 24) & 0xFF);
}
static void write_le_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
}

// makes file header for .wav file https://docs.fileformat.com/audio/wav/
static void fill_wav_header(uint8_t *hdr, uint32_t sample_rate, uint16_t bits_per_sample,
                            uint16_t channels, uint32_t data_bytes) {
    // riff
    memcpy(hdr + 0, "RIFF", 4);
    // file size = 36 + data_bytes
    write_le_u32(hdr + 4, 36 + data_bytes);
    // WAVE
    memcpy(hdr + 8, "WAVE", 4);

    // fmt chunk
    memcpy(hdr + 12, "fmt ", 4);
    write_le_u32(hdr + 16, 16); // fmt chunk size
    write_le_u16(hdr + 20, 1); // audio format = 1 (PCM)
    write_le_u16(hdr + 22, channels); // num channels
    write_le_u32(hdr + 24, sample_rate); // sample rate
    uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
    write_le_u32(hdr + 28, byte_rate);
    uint16_t block_align = channels * (bits_per_sample / 8);
    write_le_u16(hdr + 32, block_align);
    write_le_u16(hdr + 34, bits_per_sample);

    // data chunk header
    memcpy(hdr + 36, "data", 4);
    write_le_u32(hdr + 40, data_bytes);
}

static void log_time(int command) { // if command = 0, log the start time; if command = 1, log the end time
	if (command == 0) {
		cur_time_starts = get_seconds();
		return;
	} else if (command != 1) {
		print("Error wrong code\n\r");
		return;
	}

	cur_time_ends = get_seconds();

	FIL log_file;
	FRESULT rc = f_open(&log_file, "timelog.txt", FA_OPEN_APPEND | FA_WRITE);
	if (rc != FR_OK) {
		print("Failed to open logfile\n");
		return;
	}

	f_lseek(&log_file, f_size(&log_file)); // go to the end of file if it exists

	int start_sec = (int)cur_time_starts;
	int end_sec = (int)cur_time_ends;
	int len_sec = end_sec - start_sec;
	char time_display_start[9];
	char time_display_end[9];
	char recording_total_time[9];
	format_hms(start_sec, time_display_start);
	format_hms(end_sec, time_display_end);
	format_hms(len_sec, recording_total_time);

	char buf[100];
	int len = snprintf(buf, sizeof(buf), "Recording %d logged:\nStart: %s\nEnd: %s\nLength: %s\r\n\r\n",
		recordings_counter, time_display_start, time_display_end, recording_total_time
	);
	UINT writes;
	f_write(&log_file, buf, len, &writes);
	f_close(&log_file);

	recordings_counter++;

	return;

}

static int get_seconds() {
    XTime time;
    XTime_GetTime(&time);
    double seconds = (double)time / (double)COUNTS_PER_SECOND;
    return (int)seconds;
}

// convert seconds to hms
static void format_hms(int seconds, char *buf) {
    int h = seconds / 3600;
    int m = (seconds % 3600) / 60;
    int s = seconds % 60;
    snprintf(buf, 16, "%02d:%02d:%02d", h, m, s);
}

static void wipe_log_file() {
	FRESULT fres = f_mount(&FatFs, "", 1);
	    if (fres != FR_OK) {
	        xil_printf("f_mount failed: %d\r\n", fres);
	        return;
	    } else {
	        xil_printf("SD card mounted\r\n");
	    }

	FIL log_file;
	fres = f_open(&log_file, "timelog.txt", FA_CREATE_ALWAYS | FA_WRITE);
	if (fres == FR_OK) {
		f_close(&log_file);
		xil_printf("Log file cleared at startup.\r\n");
	} else {
		xil_printf("Failed to clear log file: %d\r\n", fres);
	}

	f_close(&log_file);
	return;
}
